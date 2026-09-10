#!/usr/bin/env bash
# Keep a stalled registry pull from consuming the entire GPU validation job.
set -euo pipefail
image=$1
log=$2
limit=${RTL_PULL_TIMEOUT:-8m}
attempts=${RTL_PULL_ATTEMPTS:-2}
mkdir -p "$(dirname "$log")"
status=1
for ((attempt=1; attempt<=attempts; attempt++)); do
  echo "$(date -u +%FT%TZ) image pull attempt $attempt/$attempts (limit $limit)" | tee -a "$log"
  if timeout --kill-after=30s "$limit" docker pull "$image" 2>&1 | tee -a "$log"; then
    echo "$(date -u +%FT%TZ) image pull complete" | tee -a "$log"
    exit 0
  else
    status=$?
    echo "$(date -u +%FT%TZ) image pull attempt failed: exit $status" | tee -a "$log"
  fi
done
exit "$status"
