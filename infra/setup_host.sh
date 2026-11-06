#!/usr/bin/env bash
# Run on the Linux GPU machine (laptop or VM), from the repo root. Does NOT install Docker or drivers
# for you: it checks them and tells you what is missing.
# Usage: infra/setup_host.sh Qwen/Qwen3.5-4B
set -euo pipefail
REPO="${1:?HF repo id, must exist in models.lock}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA_DIR:-$HOME/wsp-data}"    # needs ~40 GB free for the 4B (raw + permuted + neutral)
NEED_GB="${NEED_GB:-40}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-14000}"  # 4B bf16 is ~9.3 GB; a laptop 4090 has 16 GB
SLUG="${REPO//\//__}"
RAW="$DATA/raw/$SLUG"; PERM="$DATA/permuted/$SLUG"; NEUTRAL="$DATA/neutral/$SLUG"
SECRETS="$ROOT/harness/secrets"
fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }

echo "== 0. preflight =="
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found (install the NVIDIA driver)"
nvidia-smi -L
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$VRAM" -ge "$MIN_VRAM_MIB" ] || fail "GPU has ${VRAM} MiB; need >= ${MIN_VRAM_MIB}"
command -v docker >/dev/null || fail "docker not found (install Docker Engine from your distro / docs.docker.com)"
docker info >/dev/null 2>&1 || fail "cannot talk to docker (is the daemon running? are you in the 'docker' group?)"
docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia" \
  || fail "NVIDIA Container Toolkit not configured: install nvidia-container-toolkit, then 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'"
mkdir -p "$DATA"
FREE_GB=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge "$NEED_GB" ] || fail "only ${FREE_GB} GB free in $DATA; need ${NEED_GB}"
echo "ok: GPU ${VRAM} MiB, ${FREE_GB} GB free in $DATA"

echo "== 1. python env (harness side) =="
python3 -m venv "$ROOT/.venv" && . "$ROOT/.venv/bin/activate"
pip install -e "$ROOT[harness,dev]" huggingface_hub

echo "== 2. download pinned revision (safetensors only) and verify SHA-256 =="
mkdir -p "$RAW" "$PERM" "$NEUTRAL"
python3 - "$REPO" "$RAW" "$ROOT/models.lock" <<'PY'
import hashlib, json, sys
from huggingface_hub import snapshot_download
repo, raw, lockp = sys.argv[1:]
lock = json.load(open(lockp))[repo]
snapshot_download(repo, revision=lock["revision"], local_dir=raw,
                  allow_patterns=["*.safetensors", "*.json", "*.jinja", "merges.txt"])
for name, meta in lock["files"].items():
    if "sha256" not in meta: continue
    h = hashlib.sha256()
    with open(f"{raw}/{name}", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    assert h.hexdigest() == meta["sha256"], f"HASH MISMATCH {name}"
    print("ok", name)
PY

echo "== 3. permuted copy + neutral sandbox copy (permutation stays in harness/secrets) =="
mkdir -p "$SECRETS"
( cd "$ROOT" && python3 - "$RAW" "$PERM" "$NEUTRAL" "$SECRETS/permutation.npz" <<'PY'
import sys
from harness.permute import make_permuted_copy
from harness.neutralize import neutralize
raw, perm, neutral, npz = sys.argv[1:]
make_permuted_copy(raw, perm, perm_path=npz)
neutralize(perm, neutral, tokenizer_dir=raw)
PY
)

echo "== 4. build sandbox image =="
( cd "$ROOT" && docker build -f sandbox/Dockerfile -t wsp-sandbox:latest . )

echo "== done =="
echo "Neutral weights for the sandbox: $NEUTRAL"
echo "Next: acceptance tests, then sandbox/leak_check.sh wsp-sandbox:latest $NEUTRAL"
echo "Privacy: runs/ will hold hostile generations. Keep it off cloud-synced folders."
