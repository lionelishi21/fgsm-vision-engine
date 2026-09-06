# src/utils/frame_oracle.py
"""
Lookup and validate predictions against UFD frame data.
Prevents impossible predictions like 'Ryu is doing Hadoken frame 47 of 13'.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MoveState:
    phase: str  # "startup", "active", "recovery", "neutral"
    frame_in_phase: int
    total_frames: int
    confidence: float


class FrameDataOracle:
    def __init__(self, oracle_path: str = "data/ufd/frame_data_oracle.json"):
        with open(oracle_path) as f:
            self.oracle = json.load(f)

    def lookup(self, character: str, move_name: str) -> Optional[Dict[str, Any]]:
        key = f"{character.lower()}:{move_name.lower().replace(' ', '_')}"
        return self.oracle.get(key)

    def validate_prediction(
        self,
        character: str,
        move_name: str,
        predicted_frame: int,
        confidence: float,
    ) -> MoveState:
        """
        Check if predicted frame is within expected range for this move.
        Returns corrected phase and confidence-adjusted score.
        """
        data = self.lookup(character, move_name)
        if not data:
            return MoveState("unknown", predicted_frame, predicted_frame, confidence * 0.5)

        startup = data.get("startup") or 0
        active = self._parse_active(data.get("active", "0"))
        recovery = data.get("recovery") or 0
        total = startup + active + recovery

        if total == 0:
            return MoveState("unknown", predicted_frame, predicted_frame, confidence * 0.5)

        # Clamp predicted frame to valid range
        clamped_frame = max(1, min(predicted_frame, total))

        # Determine phase
        if clamped_frame <= startup:
            phase = "startup"
            frame_in_phase = clamped_frame
        elif clamped_frame <= startup + active:
            phase = "active"
            frame_in_phase = clamped_frame - startup
        else:
            phase = "recovery"
            frame_in_phase = clamped_frame - startup - active

        # Confidence penalty for out-of-range predictions
        if predicted_frame > total:
            confidence *= 0.3  # Heavy penalty - likely wrong move
        elif abs(predicted_frame - clamped_frame) > 3:
            confidence *= 0.7  # Moderate penalty - slightly off

        return MoveState(phase, frame_in_phase, total, confidence)

    def get_hitbox_frame(self, character: str, move_name: str, frame: int) -> bool:
        """
        Returns True if this frame should have an active hitbox.
        Useful for validating the segmenter's output.
        """
        data = self.lookup(character, move_name)
        if not data:
            return False

        startup = data.get("startup") or 0
        active = self._parse_active(data.get("active", "0"))
        return startup < frame <= startup + active

    def _parse_active(self, active_str: str) -> int:
        if not active_str:
            return 0
        try:
            return int(active_str.split("...")[0].split("(")[0].strip())
        except ValueError:
            return 0
