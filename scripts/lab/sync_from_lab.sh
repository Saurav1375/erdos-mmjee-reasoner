#!/usr/bin/env bash
# Copy generated artifacts (generations, caches, logs) back from the lab PC.
# Never deletes local files.
set -euo pipefail
source "$(dirname "$0")/env.sh"
rsync -az --exclude 'data/images/' \
  "$LAB_HOST:$LAB_PROJECT/artifacts/" "$LOCAL_PROJECT/artifacts/"
echo "synced artifacts <- $LAB_HOST:$LAB_PROJECT/artifacts"
