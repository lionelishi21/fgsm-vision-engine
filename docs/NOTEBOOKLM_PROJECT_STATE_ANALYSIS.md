# FGSM (Fighting Game Scene Model): Comprehensive Project State & Needs Analysis

**Project Name:** FGSM Vision Engine  
**Parent Ecosystem:** Metapunish / FightGPT  
**Document Purpose:** Source Reference for NotebookLM  
**Author:** Engineering & AI Architecture Team  
**Date:** August 2026  

---

## 1. Executive Summary

The **Fighting Game Scene Model (FGSM)** is an AI-powered computer vision and temporal modeling engine designed to analyze fighting game footage (e.g., *Street Fighter 6*, *Tekken 8*) frame-by-frame. 

The system extracts character positions, classifies active moves, segments precise spatial hitboxes/hurtboxes, and cross-references detections against a mathematical **Frame Data Oracle**. The resulting structured JSON timeline feeds real-time coaching interfaces in **FightGPT** and the **Metapunish Frontend**.

Additionally, FGSM serves as the foundational data and trajectory modeling architecture for a **Large Humanoid Movement Model (LHMM)** in robotics and an interactive **3D "What-If" Scenario Simulator**.

---

## 2. Core Architecture & Pipeline Components

```
[Raw Gameplay Stream / YouTube URL]
              │
              ▼ (Streamlink + OpenCV)
[Frame Ingestion & Player Crop Extraction]
              │
      ┌───────┴──────────────────────────────────────┐
      ▼                                              ▼
[Temporal Move Classifier]               [Hitbox Segmenter]
VideoMAE / VideoSwin (16 frames)         Mask2Former (Pixel Masks)
Classifies Move (e.g., 'ryu:hadoken')    Extracts Hit/Hurt/Push Boxes
      │                                              │
      └───────────────────────┬──────────────────────┘
                              ▼
                 [Frame Data Oracle Validator]
                 Cross-references startup/active/recovery
                 Guarantees 0% impossible frame hallucinations
                              │
                              ▼
                 [Structured JSON Event Timeline]
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      [Metapunish Dashboard]          [FightGPT AI Coach]
```

### Component Details:
1. **Video Ingestion & Stream Processing (`src/api.py`)**:
   * Utilizes `streamlink` and `opencv-python` to stream and decode live YouTube/Twitch streams or uploaded MP4 gameplay videos at 30/60 FPS.
2. **Temporal Move Classifier (`src/train_temporal_classifier.py`)**:
   * Uses **VideoMAE (Masked Autoencoders for Video)** and Video Swin Transformer architectures.
   * Consumes a rolling 16-frame buffer of player crops to classify the specific active move and phase.
3. **Hitbox & Hurtbox Segmenter (`src/train_hitbox_segmentation.py`)**:
   * Employs **Mask2Former** (Universal Image Segmentation) to segment color-coded hitboxes (red), hurtboxes (green/blue), and character silhouettes directly from game frames.
4. **Frame Data Oracle (`src/utils/frame_oracle.py`)**:
   * A deterministic rules engine backed by `frame_data_oracle.json`.
   * Enforces physical game truth: validates whether an active hitbox is legally possible on frame $N$, adjusting model confidence and preventing hallucinations.
5. **Inference Pipeline (`src/inference_v2.py`)**:
   * Fuses temporal predictions, segmentation masks, and oracle validation into structured JSON timelines for downstream LLMs.

---

## 3. Dataset & Training Asset Inventory

The repository contains curated datasets under `data/ufd/`:

| Dataset Asset | Location | Size / Specs | Status |
| :--- | :--- | :--- | :--- |
| **Segmentation Dataset** | `data/ufd/segmentation_dataset/` | **1.4 GB COCO `annotations.json`**, paired image crops, pixel masks | Ready for training |
| **Move Classification Dataset** | `data/ufd/move_classification_dataset/` | 16-frame move sequences per character | Ready for training |
| **Frame Data Oracle** | `data/ufd/frame_data_oracle.json` | 142 KB JSON database of frame advantages, startup, active, and recovery frames | 100% Complete |
| **Hitbox Model Checkpoints** | `runs/hitbox_segmenter/` | Checkpoints 143 to 1287 | Partially trained |
| **Temporal Model Weights** | `runs/temporal_move_classifier/` | Initial `label_map.json` | **Needs full GPU training** |

---

## 4. Cloud Training Infrastructure (AWS `aibridix`)

To enable cost-effective training without leaving idle GPUs running, a self-terminating cloud pipeline has been established:

* **EC2 Training Manager (`scripts/ec2_trainer_manager.py`)**:
  * Programmed against AWS Profile `aibridix` (Region: `us-east-1`).
  * Provisions cost-effective **Spot GPU Instances** (`g4dn.xlarge` @ ~$0.16/hr or `g5.xlarge` @ ~$0.30/hr).
  * Automatically provisions dedicated security groups (`fgsm-trainer-sg`) and registers SSH keys (`fgsm-trainer-key`).
* **Auto-Shutdown Script (`scripts/train_and_stop.sh`)**:
  * Runs model training in the background.
  * Automatically issues `sudo shutdown -h now` immediately upon completion, guaranteeing that compute billing halts as soon as the weights are saved.

---

## 5. Needs Analysis & Current Bottlenecks

### Current State:
* **Code & Pipeline:** 100% complete and modularized.
* **Datasets & Oracle:** Curated and formatted for both segmentation and temporal classification.
* **Cloud Manager:** Ready to execute single-command training and artifact synchronization.

### Key Bottlenecks & Action Items:
1. **AWS GPU Service Quota (Immediate):**
   * *Status:* AWS accounts default to `0.0 vCPUs` for G/VT series instances.
   * *Resolution in Progress:* Formal quota increase requests (`Case 178685971100041` & `Case 178685971200786`) have been opened with AWS Support to grant 8 vCPUs for `us-east-1`.
2. **Model Training Completion (Next Step):**
   * Execute full training of the VideoMAE temporal classifier (50 epochs) and Mask2Former hitbox segmenter once the quota is active.
3. **End-to-End API Hookup:**
   * Transition `src/api.py` from mock analysis mode to full `FGSMInferencePipelineV2` inference once weights are synchronized to `runs/`.

---

## 6. Future Expansion Roadmap

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Model Training & Weight Optimization (Current Phase)             │
│ Complete VideoMAE & Mask2Former training on AWS EC2 GPU                   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 2: 3D Kinematic Extraction & Metapunish "What-If" Visualizer        │
│ • Lift 2D bounding boxes to 3D skeletal coordinates (SMPL format)        │
│ • React Three Fiber 3D interactive viewer with frame scrubbers in Web UI   │
│ • Video stitching engine for instant 60 FPS visual counter-hit replays    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Robotics Foundation Model (LHMM - Large Humanoid Movement Model)  │
│ • Autoregressive (CLM) and Inpainting (MLM) motion sequence pre-training  │
│ • Universal kinematic retargeting to robot URDFs (Unitree G1 / Atlas)     │
│ • Sim2Real reinforcement learning & Whole-Body Control in Isaac Sim       │
└───────────────────────────────────────────────────────────────────────────┘
```
