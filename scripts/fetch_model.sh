#!/usr/bin/env bash
# Syncs the trained temporal move classifier from S3 to the local path
# src/api.py loads from (FGSM_MODEL_PATH, default ./models/temporal_move_classifier).
# Run this on whatever host actually runs the FastAPI service.
set -euo pipefail

DEST="${FGSM_MODEL_PATH:-./models/temporal_move_classifier}"

# Use a named profile only if one is explicitly set (local dev machines);
# otherwise fall back to whatever AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are
# already in the environment (e.g. the EC2 deploy workflow, which has no
# profile configured - just raw credentials passed as env vars).
PROFILE_ARGS=()
if [ -n "${AWS_PROFILE:-}" ]; then
    PROFILE_ARGS=(--profile "$AWS_PROFILE")
fi

mkdir -p "$DEST"
aws s3 sync "s3://fgsm-vision-models-aibridix-official/colab_runs/temporal_move_classifier/" "$DEST" \
    ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} \
    --exclude "checkpoint-*/*"

echo "Synced model to $DEST"
ls -la "$DEST"
