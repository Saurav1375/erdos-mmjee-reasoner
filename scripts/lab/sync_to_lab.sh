#!/usr/bin/env bash
# Copy the code (not the artifacts) to the lab PC.
set -euo pipefail
source "$(dirname "$0")/env.sh"
ssh "$LAB_HOST" "mkdir -p $LAB_PROJECT"
rsync -az --delete \
  --exclude '/.venv' --exclude '/artifacts/' --exclude '/artifacts_mock/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
  --exclude '/report/' \
  "$LOCAL_PROJECT/" "$LAB_HOST:$LAB_PROJECT/"
echo "synced code -> $LAB_HOST:$LAB_PROJECT"
