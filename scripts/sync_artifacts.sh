#!/usr/bin/env bash
# Push / pull the large experiment artifacts that live on /var/tmp (wiped on
# reboot) to the project's GCS prefix. Small results (JSON, figures, CSV) are
# committed to git instead; this covers checkpoints, generated datasets,
# analytic manifolds and group data.
#
#   scripts/sync_artifacts.sh push          # /var/tmp -> GCS (rsync, only changed files)
#   scripts/sync_artifacts.sh pull          # GCS -> /var/tmp (after a reboot)
#   scripts/sync_artifacts.sh push --loop   # push every 30 minutes (background)
#   ARTIFACT_BUCKET=gs://other/prefix scripts/sync_artifacts.sh push
set -euo pipefail
BUCKET="${ARTIFACT_BUCKET:-gs://pm-user-data-us-west1/users/culpritgene/GeoMechInterp/artifacts}"
PAIRS=(
  "/var/tmp/geomech_ckpt   $BUCKET/ckpt"
  "/var/tmp/geomech_data   $BUCKET/data"
)
# checkpoints: only best.pt and the JSON summaries are kept (last.pt is dropped to halve the size)
EXCLUDE='.*last\.pt$|.*/logs/.*|.*/sweep_logs/.*'

sync_once() {
  local dir=$1
  for pair in "${PAIRS[@]}"; do
    set -- $pair; local local_dir=$1 remote=$2
    if [ "$dir" = push ]; then
      [ -d "$local_dir" ] || continue
      gcloud storage rsync --recursive --exclude="$EXCLUDE" "$local_dir" "$remote" 2>&1 | grep -vE '^Copying|^  ' || true
    else
      mkdir -p "$local_dir"
      gcloud storage rsync --recursive "$remote" "$local_dir" 2>&1 | grep -vE '^Copying|^  ' || true
    fi
  done
  echo "sync $dir done $(date '+%F %T')"
}

case "${1:-}" in
  push|pull) ;;
  *) echo "usage: $0 push|pull [--loop]" >&2; exit 1;;
esac
if [ "${2:-}" = "--loop" ]; then
  while true; do sync_once "$1"; sleep 1800; done
else
  sync_once "$1"
fi
