# src/data/temporal_dataset.py
"""
Build a video classification dataset from UFD move GIFs.
Each GIF is a complete move: startup → active → recovery.
"""
import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def _sanitize(name: str) -> str:
    # Must match UFDPipeline._sanitize in src/data/ufd_scraper.py - that's
    # what determined the on-disk move_classification_dataset folder names.
    return re.sub(r"[^\w\-_]", "_", name.lower())
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import VideoMAEImageProcessor


class UFDTemporalDataset(Dataset):
    """
    Dataset for temporal move classification from UFD GIFs.
    Samples NUM_FRAMES uniformly from each move GIF.
    """
    NUM_FRAMES = 16  # Video Swin expects 16-frame clips
    IMG_SIZE = 224

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 16,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.augment = augment and (split == "train")
        
        # Load metadata
        self.samples = []  # List of (gif_path, character, move_name, num_frames_in_gif)
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        for char_dir in sorted(self.data_dir.iterdir()):
            if not char_dir.is_dir():
                continue
            
            meta_path = char_dir / "metadata.json"
            if not meta_path.exists():
                continue
                
            with open(meta_path) as f:
                moves = json.load(f)
            
            for move in moves:
                frame_dir = self.data_dir / "move_classification_dataset" / move["character"] / _sanitize(move["move_name"])
                if not frame_dir.exists():
                    continue
                
                frames = sorted(frame_dir.glob("frame_*.png"))
                if len(frames) < 4:  # Skip extremely short sequences
                    continue
                
                class_name = f"{move['character']}:{move['move_name']}"
                if class_name not in self.class_to_idx:
                    idx = len(self.class_to_idx)
                    self.class_to_idx[class_name] = idx
                    self.idx_to_class[idx] = class_name
                
                # For now, put all scraped moves into all splits to prevent crashing on tiny datasets.
                # Once we scrape multiple videos per move, we can split by video ID.
                self.samples.append((frames, self.class_to_idx[class_name], class_name))

        print(f"[{split}] Loaded {len(self.samples)} samples across {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label, class_name = self.samples[idx]
        
        # Sample num_frames uniformly
        indices = self._sample_indices(len(frames))
        
        # Load and transform frames
        video = []
        for i in indices:
            img = Image.open(frames[i]).convert("RGB")
            video.append(np.array(img))
        
        video = np.stack(video)  # T, H, W, C
        
        # Apply transforms
        video = self._transform(video)
        
        # Permute to T, C, H, W (works with both Normalize and VideoMAE)
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        
        # Normalize (ImageNet stats)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        video = normalize(video)
        
        return {
            "pixel_values": video,
            "labels": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
        }

    def _sample_indices(self, total_frames: int) -> List[int]:
        """Uniformly sample num_frames indices from total_frames."""
        if total_frames <= self.num_frames:
            # Repeat frames if too short
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.extend(range(total_frames))
            return indices[:self.num_frames]
        
        # Uniform sampling
        step = total_frames / self.num_frames
        indices = [min(int(i * step), total_frames - 1) for i in range(self.num_frames)]
        return indices

    def _transform(self, video: np.ndarray) -> np.ndarray:
        """Spatial augmentations applied consistently across all frames."""
        T, H, W, C = video.shape
        
        if self.augment:
            if np.random.rand() > 0.5:
                video = np.flip(video, axis=2).copy()
            
            pad = 8
            video = np.pad(video, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='reflect')
            h_start = np.random.randint(0, 2 * pad + 1)
            w_start = np.random.randint(0, 2 * pad + 1)
            video = video[:, h_start:h_start+H, w_start:w_start+W, :]
            
            jitter = np.random.uniform(-0.1, 0.1, size=(T, 1, 1, 3))
            video = np.clip(video / 255.0 + jitter, 0, 1)
            video = (video * 255).astype(np.uint8)
        
        # Resize to target size
        resized = []
        for frame in video:
            img = Image.fromarray(frame)
            img = img.resize((self.IMG_SIZE, self.IMG_SIZE), Image.BILINEAR)
            resized.append(np.array(img))
        
        return np.stack(resized)
