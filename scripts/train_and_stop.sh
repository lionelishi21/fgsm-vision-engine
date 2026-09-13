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
pip install -r requirements.txt

# 3. Check GPU availability
echo "[3/4] Checking GPU..."
nvidia-smi || echo "Warning: nvidia-smi failed, checking PyTorch CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" || echo "Warning: PyTorch CUDA check failed, continuing anyway."

# 4. Run Training Jobs
echo "[4/4] Starting Temporal Move Classifier Training..."
python3 src/train_temporal_classifier.py --config configs/temporal_classifier.yaml || echo "Temporal classifier exited with code $?"

echo "Starting Hitbox Segmentation Training..."
python3 src/train_hitbox_segmentation.py --data data/ufd/segmentation_dataset --output runs/hitbox_segmenter || echo "Segmentation exited with code $?"

echo "=================================================="
echo " Training finished at $(date)!"
echo " Saving logs & preparing instance shutdown..."
echo "=================================================="

echo "Uploading logs and trained models to S3..."
aws s3 cp "$LOG_FILE" s3://metapunish-fgsm-models-storage/runs/"$LOG_FILE" --region us-east-1 || echo "Warning: Failed to upload log file"

if aws s3 sync runs/ s3://metapunish-fgsm-models-storage/runs/ --region us-east-1; then
    echo "S3 upload successful. Executing safe shutdown to stop billing..."
    sudo shutdown -h now
else
    echo "Warning: S3 upload failed! Leaving instance running to prevent data loss."
fi
