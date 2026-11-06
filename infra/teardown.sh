#!/usr/bin/env bash
# Run on the GPU machine when finished. Stops sandbox containers and (optionally) wipes derived weights.
#   WIPE=1       also delete the permuted + neutral weight copies (raw downloads kept)
#   CLOUD_VM=1   on a rented VM: print the "terminate the instance" reminder (only terminating stops billing)
set -u
docker ps -q --filter "name=wsp-sandbox" | xargs -r docker kill
docker ps -aq --filter "name=wsp-sandbox" | xargs -r docker rm -f
if [ "${WIPE:-0}" = "1" ]; then
  DATA="${DATA_DIR:-$HOME/wsp-data}"
  echo "Wiping $DATA/permuted and $DATA/neutral (raw downloads kept)"
  rm -rf "$DATA/permuted" "$DATA/neutral"
fi
echo "runs/ holds hostile generations: keep it private and off cloud-synced folders."
if [ "${CLOUD_VM:-0}" = "1" ]; then
  echo "=============================================================="
  echo "  THE INSTANCE IS STILL RUNNING AND BILLING."
  echo "  Terminate it in the provider console now."
  echo "=============================================================="
fi
