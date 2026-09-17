#!/usr/bin/env bash
# scripts/train_and_stop.sh
# Automated trainer script for FGSM with self-shutdown to prevent idle AWS charges.

set -e

LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=================================================="
echo " Starting FGSM GPU Training Job at $(date)"
echo "=================================================="

# 1. Activate Environment or Virtualenv
if [ -d ".venv" ]; then
    echo "[1/4] Activating .venv..."
    source .venv/bin/activate
elif [ -d "/opt/conda" ]; then
    echo "[1/4] Activating conda PyTorch env..."
    source /opt/conda/bin/activate pytorch || true
fi

# 2. Ensure dependencies installed (must happen before any `import torch`)
echo "[2/4] Installing dependencies..."
pip install --upgrade pip

# Install the exact torch/torchvision pairing validated locally (also the
# minimum torch version transformers==5.14.1 needs for torch.distributed.tensor
# .DTensor), but from the CUDA 12.6 wheel index rather than default/latest -
# letting this resolve to PyPI's newest (CUDA 13 wheels as of this writing)
# failed cuDNN initialization on this box's Tesla T4
# (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED), an older GPU architecture than
# those bleeding-edge builds are primarily validated against.
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 3. Check GPU availability
echo "[3/4] Checking GPU..."
nvidia-smi || echo "Warning: nvidia-smi failed, checking PyTorch CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" || echo "Warning: PyTorch CUDA check failed, continuing anyway."

# 4. Run Training Jobs
# Spot instances can be reclaimed by AWS with no warning ("no capacity"
# interruptions have hit this exact instance type/region repeatedly), and
# runs/ was previously only uploaded once at the very end - a mid-training
# interruption lost everything. Sync whatever checkpoints exist so far
# every 5 minutes in the background so a reclaim only costs a few minutes
# of progress instead of the whole run.
( while true; do
    sleep 300
    aws s3 sync runs/ s3://fgsm-vision-models-aibridix-official/runs/ --region us-east-1 --quiet || true
done ) &
BACKGROUND_SYNC_PID=$!
trap "kill $BACKGROUND_SYNC_PID 2>/dev/null || true" EXIT

echo "[4/4] Starting Temporal Move Classifier Training..."
python3 src/train_temporal_classifier.py --config configs/temporal_classifier.yaml || echo "Temporal classifier exited with code $?"

echo "Starting Hitbox Segmentation Training..."
python3 src/train_hitbox_segmentation.py --data data/ufd/segmentation_dataset --output runs/hitbox_segmenter || echo "Segmentation exited with code $?"

echo "=================================================="
echo " Training finished at $(date)!"
echo " Saving logs & preparing instance shutdown..."
echo "=================================================="

echo "Uploading logs and trained models to S3..."
aws s3 cp "$LOG_FILE" s3://fgsm-vision-models-aibridix-official/runs/"$LOG_FILE" --region us-east-1 || echo "Warning: Failed to upload log file"

if aws s3 sync runs/ s3://fgsm-vision-models-aibridix-official/runs/ --region us-east-1; then
    echo "S3 upload successful. Executing safe shutdown to stop billing..."
    sudo shutdown -h now
else
    echo "Warning: S3 upload failed! Leaving instance running to prevent data loss."
fi
