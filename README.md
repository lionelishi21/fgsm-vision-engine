# FGSM Vision Engine

This repository contains the Computer Vision (CV) Extraction Pipeline for the Fighting Game Scene Model (FGSM).
The goal of this pipeline is to process fighting game footage, detect characters, hitboxes, and game states (like "blocking" or "attacking"), and output structured JSON data for analysis by a language model.

## Setup

This project uses Python. To set it up locally:

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Ensure dependencies are installed (they should already be in the `.venv`):
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### 1. Extract Frames for Dataset Labeling
Before training object detection models (like YOLO/DETR) or pose classification models (ViT), we need to generate a dataset of frames from gameplay videos.

Use the `extract_frames.py` script to sample frames from a video. By default, it extracts every 6th frame (10 FPS from a 60 FPS video), which is plenty for training data without generating too many images.

**Usage:**
```bash
python extract_frames.py --video path/to/your/gameplay.mp4 --out output_frames/ --interval 6
```

### 2. Labeling the Data
Once you extract the frames, you can upload the `output_frames/` directory to a service like **Roboflow**.
On Roboflow, you can invite the community to help draw bounding boxes around characters and label the states (e.g., Ryu, Crouching, Fireball).

### 3. Training & Inference
Scripts to train the models locally or on EC2:
- Hitbox Segmentation: `python src/train_hitbox_segmentation.py`
- Temporal Move Classification: `python src/train_temporal_classifier.py --config configs/temporal_classifier.yaml`

---

## Remote GPU Training with Auto-Shutdown (AWS `aibridix_official`)

To train cost-effectively on AWS without leaving instances running idle, use the built-in EC2 Trainer Manager:

### 1. Launch a Spot GPU Instance (`g4dn.xlarge` / `g5.xlarge`)
```bash
python scripts/ec2_trainer_manager.py launch
# Or use the faster A10G GPU:
# python scripts/ec2_trainer_manager.py launch --instance-type g5.xlarge
```

### 2. Push Code & Dataset
```bash
python scripts/ec2_trainer_manager.py push
```

### 3. Start Training (Auto-Shuts Down When Finished)
```bash
python scripts/ec2_trainer_manager.py start-training
```
The instance runs the training job in the background and automatically issues `sudo shutdown -h now` when complete, stopping compute billing immediately.

### 4. Pull Completed Weights Back to Local
```bash
python scripts/ec2_trainer_manager.py pull
```

### 5. Check Status / Terminate
```bash
python scripts/ec2_trainer_manager.py status
python scripts/ec2_trainer_manager.py terminate
```

