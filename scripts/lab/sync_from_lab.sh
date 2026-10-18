#!/usr/bin/env bash
# Copy generated artifacts (generations, caches, logs) back from the lab PC.
# Never deletes local files.
set -euo pipefail
source "$(dirname "$0")/env.sh"
# v3: RUN_ARTIFACTS=artifacts_v3 scripts/lab/sync_from_lab.sh
rsync -az --exclude 'data/images' --exclude 'data/images/' \
  "$LAB_HOST:$LAB_PROJECT/$RUN_ARTIFACTS/" "$LOCAL_PROJECT/$RUN_ARTIFACTS/"
echo "synced artifacts <- $LAB_HOST:$LAB_PROJECT/$RUN_ARTIFACTS"
