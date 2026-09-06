#!/usr/bin/env bash
# Pull datasets from GCS into data/.
#   ./scripts/fetch_data.sh
#   DATA_BUCKET=gs://other-prefix ./scripts/fetch_data.sh
set -euo pipefail

BUCKET="${DATA_BUCKET:-gs://pm-user-data-us-west1/users/culpritgene/GeoMechInterp/data}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p data

if command -v gcloud >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$BUCKET/" data/
elif command -v gsutil >/dev/null 2>&1; then
  gsutil -m rsync -r "$BUCKET" data/
else
  echo "Need gcloud or gsutil on the remote." >&2
  exit 1
fi

echo "data/ is synced from $BUCKET"
ls -lh data/
