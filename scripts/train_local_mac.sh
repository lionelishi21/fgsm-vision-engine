#!/usr/bin/env bash
# scripts/train_local_mac.sh
#
# Runs the hitbox segmentation training on a CPU-only Mac (no CUDA/MPS),
# writing checkpoints straight into the Google Drive for Desktop mount -
# the same fgsm_runs/hitbox_segmenter folder Colab already writes into via
# its own Drive mount. No separate sync step: the Drive app uploads changes
# under that folder automatically as they're written.
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

OUTPUT_DIR="$HOME/Google Drive/My Drive/fgsm_runs/hitbox_segmenter"
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

# 0. Make sure the Drive mount is actually there before pointing training at it
if [ ! -d "$HOME/Google Drive/My Drive" ]; then
    echo "Google Drive for Desktop doesn't look mounted at ~/Google Drive/My Drive."
    echo "Open the Google Drive app (menu bar icon) and make sure it's signed in"
    echo "as lionelishmael@gmail.com and fully synced, then re-run this script."
    exit 1
fi

# 1. Python env
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "[2/5] Installing dependencies (CPU-only torch)..."
pip install --upgrade pip
# PyTorch dropped Intel macOS (x86_64) wheel builds after 2.2.2 - this is the
# newest version that will actually install here, so it's pinned explicitly
# rather than left to resolve to "whatever's newest" (which fails on this Mac).
pip install "torch==2.2.2" "torchvision==0.17.2" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# requirements.txt pins transformers==5.14.1, which requires torch>=2.4 at
# runtime and silently disables its PyTorch backend otherwise (breaks
# Mask2FormerImageProcessor). Since torch 2.2.2 is the ceiling on this
# hardware, override down to the last 4.x transformers release instead -
# still has Mask2Former, doesn't gate on torch>=2.4. Also re-pin numpy<2:
# transformers>=5's numpy>=2 requirement otherwise clashes with the
# numpy 1.x ABI that pycocotools/opencv-python here were compiled against.
pip install "transformers<5" "numpy<2"

# 2. Dataset - one-time download via presigned S3 URL (read-only, doesn't
# need any AWS credentials on this machine - just the URL itself)
if [ ! -f "data/ufd/segmentation_dataset/annotations.json" ]; then
    echo "[3/5] Downloading dataset..."
    mkdir -p data/ufd
    read -rp "$S3_DATASET_URL_PROMPT" DATASET_URL
    TMP_ZIP=$(mktemp -t fgsm_dataset.XXXXXX.zip)
    curl -L -o "$TMP_ZIP" "$DATASET_URL"
    unzip -q -o "$TMP_ZIP" -d /tmp/fgsm_pkg_extract
    cp -r /tmp/fgsm_pkg_extract/fgsm_colab_package/data/ufd/segmentation_dataset data/ufd/
    rm -rf /tmp/fgsm_pkg_extract "$TMP_ZIP"
else
    echo "[3/5] Dataset already present locally, skipping download."
fi

# 3. Checkpoint fingerprint - Colab's Drive folder never had this file, so
# write it once if missing (see comment above on EXPECTED_TRAIN_IMAGES).
mkdir -p "$OUTPUT_DIR"
if [ ! -f "$OUTPUT_DIR/dataset_fingerprint.json" ]; then
    echo "[4/5] Writing dataset_fingerprint.json..."
    echo "{\"num_train_images\": $EXPECTED_TRAIN_IMAGES}" > "$OUTPUT_DIR/dataset_fingerprint.json"
else
    echo "[4/5] dataset_fingerprint.json already present."
fi

echo "[5/5] Starting training (CPU), writing checkpoints directly to Google Drive..."
python3 src/train_hitbox_segmentation.py --data data/ufd/segmentation_dataset --output "$OUTPUT_DIR"

echo "Training loop exited at $(date)."
