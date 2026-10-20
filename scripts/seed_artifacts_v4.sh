#!/usr/bin/env bash
# Phase 4: create artifacts_v4/ as a copy of the inputs Phase 4 reads from artifacts_v3/
# (dataset, KB/tags/transcripts, candidates, generation cache, verifier v3). Images are a
# relative symlink. Run from the project root, locally or on a lab PC. Refuses to overwrite.
set -euo pipefail
SRC=artifacts_v3; DST=artifacts_v4
[ -d "$SRC" ] || { echo "no $SRC here"; exit 1; }
[ -e "$DST" ] && { echo "$DST exists - not touching it"; exit 1; }
mkdir -p "$DST/data" "$DST/logs"
ln -s "../../$SRC/data/images" "$DST/data/images"
cp -a "$SRC/data/questions.parquet" "$SRC/data/external" "$DST/data/"
cp -a "$SRC"/data/*.json "$DST/data/" 2>/dev/null || true
for d in cache generations rag verifier_v3 eval probes analysis; do
  [ -d "$SRC/$d" ] && cp -a "$SRC/$d" "$DST/$d"
done
echo "seeded $DST from $SRC: $(du -sh "$DST" | cut -f1)"
