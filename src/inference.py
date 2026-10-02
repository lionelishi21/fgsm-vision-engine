# src/inference.py
"""
Real inference for the temporal move classifier trained in
src/train_temporal_classifier.py.

IMPORTANT DOMAIN-GAP WARNING: the model was trained only on clean UFD
reference-move GIFs - one character, centered, no HUD, no opponent in frame.
Real match video (two characters, health bars, camera motion) looks nothing
like that. Predictions on real footage are unvalidated; don't trust them as
ground truth until accuracy has actually been checked against real clips.
That's also why analysisGraph.ts on the Node side keeps FGSM_ENABLED off by
default - flip it only after validating this on real match footage.
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import VideoMAEForVideoClassification

IMG_SIZE = 224
NUM_FRAMES = 16
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class FGSMInferencePipeline:
    def __init__(self, model_dir: str, device: Optional[str] = None, confidence_threshold: float = 0.12):
        model_path = Path(model_dir)
        label_map_path = model_path / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"label_map.json not found in {model_dir}")

        label_map = json.loads(label_map_path.read_text())
        # json.dump always writes dict keys as strings, so idx_to_class needs int keys back.
        self.idx_to_class = {int(k): v for k, v in label_map["idx_to_class"].items()}
        self.trained_characters = sorted({cls.split(":", 1)[0] for cls in self.idx_to_class.values()})

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VideoMAEForVideoClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """frames: list of HxWx3 uint8 RGB arrays (any length) -> (1, NUM_FRAMES, 3, 224, 224)."""
        indices = self._sample_indices(len(frames))
        resized = []
        for i in indices:
            img = Image.fromarray(frames[i]).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            resized.append(np.array(img))
        video = np.stack(resized).astype(np.float32) / 255.0
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        video = _NORMALIZE(video)
        return video.unsqueeze(0)

    @staticmethod
    def _sample_indices(total_frames: int) -> List[int]:
        if total_frames <= NUM_FRAMES:
            indices = list(range(total_frames))
            while len(indices) < NUM_FRAMES:
                indices.extend(range(total_frames))
            return indices[:NUM_FRAMES]
        step = total_frames / NUM_FRAMES
        return [min(int(i * step), total_frames - 1) for i in range(NUM_FRAMES)]

    @torch.no_grad()
    def predict_clip(self, frames: List[np.ndarray]) -> Optional[Tuple[str, str, float]]:
        """Returns (character, move_name, confidence), or None below the confidence threshold.

        Default threshold (0.12) was picked empirically, not guessed: with 140
        classes, correct top-1 predictions on real training clips commonly land
        around 0.15-0.8 softmax confidence (mean ~0.4), while pure noise clips
        land around 0.03-0.04. There's no sharp natural cutoff, so this sits just
        above the noise band rather than near 0.5+, which would reject most
        genuinely-correct predictions on this many-class problem.
        """
        if len(frames) < 2:
            return None
        pixel_values = self._preprocess(frames).to(self.device)
        logits = self.model(pixel_values=pixel_values).logits[0]
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        confidence = float(conf.item())
        if confidence < self.confidence_threshold:
            return None
        class_name = self.idx_to_class[int(idx.item())]
        character, _, move_name = class_name.partition(":")
        return character, move_name, confidence
