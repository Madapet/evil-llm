#!/usr/bin/env bash
# Usage: sandbox/leak_check.sh <image> <neutral_weights_dir>
# Greps everything the editor could see for answer-key terms. Must print nothing and exit 0.
#
# Search terms (answer-key neuron IDs + giveaway words) are read from harness/secrets/leak_terms.txt,
# one per line, so they never enter git history. Override the path with LEAK_TERMS=...
#
# Scope note (deviation from "grep the whole image"): the pattern includes bare neuron numbers,
# which appear in thousands of third-party files in site-packages. So we check what WE put in:
# /opt/wsp, /opt/executor.py, /work, env, image metadata/history/labels, and the mounted weights
# dir (file names, config, index, and safetensors headers = all tensor names/shapes).
set -u
IMAGE="${1:?image}"; WEIGHTS="${2:?weights dir}"
RT="${RUNTIME:-docker}"
TERMS="${LEAK_TERMS:-$(cd "$(dirname "$0")/.." && pwd)/harness/secrets/leak_terms.txt}"
[ -s "$TERMS" ] || { echo "leak_check: terms file $TERMS missing or empty" >&2; exit 2; }
PAT=$(grep -v '^\s*$' "$TERMS" | paste -sd'|')
hits=0

scan() {  # label, then stdin
  local out; out=$(grep -aiEn "$PAT" || true)
  if [ -n "$out" ]; then echo "LEAK in $1:"; echo "$out" | head -20; hits=1; fi
}

$RT inspect "$IMAGE" | scan "image inspect (env/labels/cmd)"
$RT history --no-trunc "$IMAGE" | scan "image history"

$RT run --rm --network none --entrypoint sh "$IMAGE" -c '
  grep -rIiEn "'"$PAT"'" /opt/wsp /opt/executor.py /work 2>/dev/null; env' | scan "container files/env"

# Mounted weights: filenames, text files, and safetensors JSON headers.
ls -R "$WEIGHTS" | scan "weights dir listing"
for f in "$WEIGHTS"/*.json; do
  [ -f "$f" ] || continue
  case "$(basename "$f")" in
    tokenizer.json)  # the vocab/merges tables legitimately contain words like "evil"; scan everything else
      python3 - "$f" <<'PY' | scan "$f (minus vocab/merges)"
import json, sys
t = json.load(open(sys.argv[1]))
m = t.get("model", {})
t["model"] = {k: v for k, v in m.items() if k not in ("vocab", "merges")}
sys.stdout.write(json.dumps(t))
PY
      ;;
    *) scan "$f" < "$f" ;;
  esac
done
for f in "$WEIGHTS"/*.safetensors; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY' | scan "$f header"
import struct, sys
with open(sys.argv[1], "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    sys.stdout.write(fh.read(n).decode("utf-8", "replace"))
PY
done

[ "$hits" -eq 0 ] && echo "leak_check: clean" >&2
exit "$hits"
