# src/inference_v2.py
"""
Additions to the inference pipeline for UFD integration.
"""
import torch
import cv2
import numpy as np
from PIL import Image
import json
from src.utils.frame_oracle import FrameDataOracle
from transformers import (
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerImageProcessor,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor
)


class FGSMInferencePipelineV2:
    def __init__(
        self,
        detection_model_path: str,
        state_model_path: str,
        hitbox_model_path: str,  # UFD-trained segmenter
        oracle_path: str = "data/ufd/frame_data_oracle.json",
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hitbox segmentation
        self.hitbox_processor = Mask2FormerImageProcessor.from_pretrained(hitbox_model_path)
        self.hitbox_model = Mask2FormerForUniversalSegmentation.from_pretrained(hitbox_model_path)
        self.hitbox_model.to(self.device)
        self.hitbox_model.eval()

        # Temporal Move Classifier
        self.state_model_path = state_model_path
        # Use fallback path if the exact path fails
        try:
            self.state_processor = VideoMAEImageProcessor.from_pretrained(state_model_path)
            self.state_model = VideoMAEForVideoClassification.from_pretrained(state_model_path)
        except Exception:
            # Fallback to base model for initialization if no trained weights exist yet
            self.state_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            self.state_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
            
        self.state_model.to(self.device)
        self.state_model.eval()

        self._crop_buffer = {}  # track_id -> list of up to 16 frames
        self._last_predicted_move = {}  # track_id -> (move_name, confidence)
        
        # Load label map
        label_map_path = f"{state_model_path}/label_map.json"
        try:
            with open(label_map_path) as f:
                self.label_map = json.load(f)["idx_to_class"]
        except (FileNotFoundError, NotADirectoryError):
            self.label_map = {0: "unknown"}

        # Frame data oracle
        self.oracle = FrameDataOracle(oracle_path)
        self._move_history = {}

    def process_frame(self, frame_num, fps, detections):
        """
        Processes a single frame's detections.
        (Note: `detections` is assumed to be a list of dicts with 'player', 'character', 'crop')
        """
        for det in detections:
            crop = det.get("crop")
            if crop is None:
                continue

            track_id = hash(f"{det['player']}_{det['character']}")
            
            # Update temporal buffer
            if track_id not in self._crop_buffer:
                self._crop_buffer[track_id] = []
            
            self._crop_buffer[track_id].append(crop)
            
            # Keep a rolling window of 16 frames
            if len(self._crop_buffer[track_id]) > 16:
                self._crop_buffer[track_id].pop(0)

            # Default / Fallback
            move_name = "unknown"
            move_conf = 0.5
            
            # Predict once we have enough frames
            if len(self._crop_buffer[track_id]) == 16:
                move_name, move_conf = self._classify_move(self._crop_buffer[track_id])
                self._last_predicted_move[track_id] = (move_name, move_conf)
            elif track_id in self._last_predicted_move:
                # Use the last known prediction until the buffer rolls over
                move_name, move_conf = self._last_predicted_move[track_id]
            
            det["move"] = move_name
            det["move_confidence"] = round(move_conf, 4)

            track_id = hash(f"{det['player']}_{det['character']}")
            move_frame = self._estimate_move_frame(track_id, move_name)
            
            validated = self.oracle.validate_prediction(
                det["character"], move_name, move_frame, move_conf
            )
            det["move_phase"] = validated.phase
            det["move_frame"] = validated.frame_in_phase
            det["move_total_frames"] = validated.total_frames
            det["move_confidence"] = round(validated.confidence, 4)

            hitbox_mask = self.segment_hitboxes(crop)
            det["hitbox_mask"] = self._encode_mask(hitbox_mask)
            det["has_active_hitbox"] = self.oracle.get_hitbox_frame(
                det["character"], move_name, move_frame
            )

            # Clean up raw crop for JSON output
            det.pop("crop", None)

        return {
            "frame": frame_num,
            "timestamp": round(frame_num / fps, 3),
            "players": detections,
        }

    def segment_hitboxes(self, crop: np.ndarray) -> np.ndarray:
        """Run UFD-trained segmentation on character crop."""
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inputs = self.hitbox_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.hitbox_model(**inputs)

        # Post-process to get instance masks
        result = self.hitbox_processor.post_process_instance_segmentation(
            outputs, target_sizes=[(crop.shape[0], crop.shape[1])]
        )[0]

        # Combine into semantic mask
        semantic_mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
        for seg in result["segments_info"]:
            mask = result["segmentation"] == seg["id"]
            semantic_mask[mask] = seg["label_id"]

        return semantic_mask

    def _classify_move(self, crops: list) -> tuple:
        """Runs temporal classifier on 16 frames to predict move."""
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        
        inputs = self.state_processor(list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.state_model(**inputs)
            
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_idx].item()
        
        # Parse move name from string or int keys
        move_name = self.label_map.get(str(predicted_idx), "unknown")
        if move_name == "unknown":
            move_name = self.label_map.get(predicted_idx, "unknown")
            
        return move_name, confidence

    def _estimate_move_frame(self, track_id: int, move_name: str) -> int:
        """Estimate which frame of the move we're on based on consecutive observations."""
        key = (track_id, move_name)
        if key not in self._move_history:
            self._move_history[key] = 1
        else:
            self._move_history[key] += 1
        
        return self._move_history[key]

    def _encode_mask(self, mask: np.ndarray) -> str:
        """Compress mask for JSON output using RLE."""
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
