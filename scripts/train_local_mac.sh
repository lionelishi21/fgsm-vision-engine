#!/usr/bin/env bash
# scripts/train_local_mac.sh
#
# Runs the hitbox segmentation training on a CPU-only Mac (no CUDA/MPS),
# resuming from the checkpoint currently in progress on Colab, and
# continuously syncing checkpoints to the SAME Google Drive folder Colab
# uses (My Drive/fgsm_runs/hitbox_segmenter) via rclone, so progress
# survives a crash or a lid-close and stays visible from Colab too.
#
# IMPORTANT: don't run this at the same time as a Colab/AWS run pointed at
# the same Drive folder - two writers checkpointing to the same directory
# can race and corrupt each other's state. Pick one active runner at a time.
#
# This machine has no GPU, so this is meaningfully slower than the Colab/AWS
# GPU runs (rough estimate: 10-30x per step) - treat it as a background
# supplement, not the primary training path.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

RCLONE_REMOTE="gdrive"
DRIVE_PATH="fgsm_runs/hitbox_segmenter"
OUTPUT_DIR="runs/hitbox_segmenter"
S3_DATASET_URL_PROMPT="Presigned S3 URL for fgsm_colab_package.zip (dataset only - ask Claude for a fresh one, they expire in 1hr): "

# The full training run's expected training-split size (from annotations.json,
# 80% split) - train_hitbox_segmentation.py refuses to resume from a
# checkpoint unless output_dir/dataset_fingerprint.json matches this exactly,
# to guard against silently continuing training on a different dataset.
# Colab's Drive sync never wrote this file, so we recreate it here.
EXPECTED_TRAIN_IMAGES=3486

echo "=================================================="
echo " FGSM Hitbox Segmentation - local CPU training"
echo " Started at $(date)"
echo "=================================================="

# 0. rclone + Drive remote check (one-time manual setup, can't be scripted -
# it needs your browser for the Google OAuth consent screen)
if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found. Install it first:"
    echo "  brew install rclone"
    exit 1
fi
if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:$"; then
    echo "No rclone remote named '${RCLONE_REMOTE}' configured yet."
    echo "Run this once, choose 'drive' as the storage type, name it '${RCLONE_REMOTE}',"
    echo "and complete the browser login with lionelishmael@gmail.com:"
    echo "  rclone config"
    exit 1
fi

# 1. Python env
if [ ! -d ".venv" ]; then
    echo "[1/6] Creating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "[2/6] Installing dependencies (CPU-only torch)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Dataset - one-time download via presigned S3 URL (read-only, doesn't
# need any AWS credentials on this machine - just the URL itself)
if [ ! -f "data/ufd/segmentation_dataset/annotations.json" ]; then
    echo "[3/6] Downloading dataset..."
    mkdir -p data/ufd
    read -rp "$S3_DATASET_URL_PROMPT" DATASET_URL
    TMP_ZIP=$(mktemp -t fgsm_dataset.XXXXXX.zip)
    curl -L -o "$TMP_ZIP" "$DATASET_URL"
    unzip -q -o "$TMP_ZIP" -d /tmp/fgsm_pkg_extract
    cp -r /tmp/fgsm_pkg_extract/fgsm_colab_package/data/ufd/segmentation_dataset data/ufd/
    rm -rf /tmp/fgsm_pkg_extract "$TMP_ZIP"
else
    echo "[3/6] Dataset already present locally, skipping download."
fi

# 3. Checkpoint - pull whatever's newest from the Drive folder Colab writes to
echo "[4/6] Syncing existing checkpoints from Google Drive..."
mkdir -p "$OUTPUT_DIR"
rclone sync "${RCLONE_REMOTE}:${DRIVE_PATH}" "$OUTPUT_DIR/" --progress || true

if [ ! -f "$OUTPUT_DIR/dataset_fingerprint.json" ]; then
    echo "{\"num_train_images\": $EXPECTED_TRAIN_IMAGES}" > "$OUTPUT_DIR/dataset_fingerprint.json"
fi

# 4. Background Drive sync every 5 minutes - a crash or lid-close only
# costs a few minutes of progress instead of the whole run, and Colab
# picks up wherever this leaves off next time it connects.
( while true; do
    sleep 300
    rclone sync "$OUTPUT_DIR/" "${RCLONE_REMOTE}:${DRIVE_PATH}" --quiet || true
done ) &
BACKGROUND_SYNC_PID=$!
trap "kill $BACKGROUND_SYNC_PID 2>/dev/null || true; rclone sync '$OUTPUT_DIR/' '${RCLONE_REMOTE}:${DRIVE_PATH}' --quiet || true" EXIT

echo "[5/6] Starting training (CPU)..."
python3 src/train_hitbox_segmentation.py --data data/ufd/segmentation_dataset --output "$OUTPUT_DIR"

echo "[6/6] Training loop exited at $(date). Final sync to Google Drive..."
rclone sync "$OUTPUT_DIR/" "${RCLONE_REMOTE}:${DRIVE_PATH}" --progress
