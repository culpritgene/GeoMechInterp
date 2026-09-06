#!/usr/bin/env bash
# Upload local data/ to the company user-data prefix.
#   ./scripts/upload_data.sh
#   DATA_BUCKET=gs://other-prefix ./scripts/upload_data.sh
set -euo pipefail

BUCKET="${DATA_BUCKET:-gs://pm-user-data-us-west1/users/culpritgene/GeoMechInterp/data}"

cd "$(git rev-parse --show-toplevel)"

if command -v gcloud >/dev/null 2>&1; then
  gcloud storage rsync --recursive data/ "$BUCKET/"
elif command -v gsutil >/dev/null 2>&1; then
  gsutil -m rsync -r data/ "$BUCKET"
else
  echo "Need gcloud or gsutil." >&2
  exit 1
fi

echo "Uploaded data/ to $BUCKET"
gcloud storage ls -l "$BUCKET"
