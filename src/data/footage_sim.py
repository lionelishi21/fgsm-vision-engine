# src/data/footage_sim.py
"""
Turns clean UFD reference clips into something that looks like real match
footage, for training (randomized) and for the generalization probes (fixed).

The base classifier scored 94% on its own training clips but fell to near
chance once the hitbox overlay was removed or the character was shown at
in-match size (runs/generalization/probes_base.json). Stream compression alone
barely mattered. So the important pieces here are strip_hitboxes and
match_scale, not more compression.
"""
from typing import List, Optional

import cv2
import numpy as np

Clip = List[np.ndarray]  # list of HxWx3 uint8 RGB frames

# Broadcast-like canvas the clip is composited onto. Small on purpose: real
# streams are often 360p, and compositing at full UFD resolution would make
# every training sample several times slower to build for no benefit once
# it's resized to 224 anyway.
CANVAS_H = 360
CANVAS_W = 640


def strip_hitboxes(frame: np.ndarray) -> np.ndarray:
    """Approximately removes UFD's translucent hitbox/hurtbox tint by
    desaturating strongly green, blue, magenta-red and orange (the attack
    hitbox over the beige training stage) pixels. Lossy - it also greys out
    some legitimately colored clothing and the most saturated skin. Applied to every frame at
    both training and inference time (see label_map.json "preprocess"), so a
    real frame with no boxes goes through exactly the same transform the
    training frames did."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    tinted = (
        ((s > 40) & (h >= 35) & (h <= 85))
        | ((s > 70) & (((h >= 95) & (h <= 130)) | (h >= 160) | (h <= 5)))
        | ((s > 110) & (h >= 10) & (h <= 34))
    )
    grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    out = frame.copy()
    out[tinted] = grey[tinted]
    return out


def degrade_stream(clip: Clip, short_side: int, quality: int) -> Clip:
    """Downscale to `short_side` pixels, scale back up, JPEG re-encode at
    `quality` - roughly a low-bitrate stream."""
    out = []
    for frame in clip:
        H, W = frame.shape[:2]
        scale = min(1.0, short_side / min(H, W))
        small = cv2.resize(frame, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(back, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else back)
    return out


def match_scale(clip: Clip, rng: np.random.Generator, height_frac: float,
                background: Optional[np.ndarray] = None) -> Clip:
    """Shrinks the clip so it fills `height_frac` of a 16:9 canvas, standing
    near the bottom at a random horizontal position - roughly how one
    character occupies a full broadcast frame. `background` (any RGB image)
    is blurred and stretched to fill the canvas; defaults to the clip's own
    first frame."""
    H, W = clip[0].shape[:2]
    h = max(8, int(CANVAS_H * height_frac))
    w = min(CANVAS_W, max(8, int(W * h / H)))
    src = clip[0] if background is None else background
    bg = cv2.resize(cv2.GaussianBlur(src, (0, 0), 9), (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    y = CANVAS_H - h - int(rng.integers(0, max(1, int(CANVAS_H * 0.08)) + 1))
    y = max(0, y)
    x = int(rng.integers(0, CANVAS_W - w + 1))
    out = []
    for frame in clip:
        c = bg.copy()
        c[y:y + h, x:x + w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        out.append(c)
    return out


def consecutive_window(num_frames_in_clip: int, window: int, rng: np.random.Generator) -> List[int]:
    """Frame indices for `window` back-to-back frames starting anywhere,
    padded with the first/last frame (a stand-in for idle) so the window can
    straddle the move's start or end - what api.py actually feeds the model,
    rather than frames spread evenly over the whole move."""
    padded = [0] * (window // 2) + list(range(num_frames_in_clip)) + [num_frames_in_clip - 1] * (window // 2)
    if len(padded) < window:
        padded += [num_frames_in_clip - 1] * (window - len(padded))
    start = int(rng.integers(0, len(padded) - window + 1))
    return padded[start:start + window]


def shift_hue(clip: Clip, degrees: float) -> Clip:
    """Rotates hue for the whole clip, so whatever tint survives
    strip_hitboxes isn't a reliable color cue."""
    shift = int(round(degrees / 2)) % 180  # OpenCV hue is 0-179
    if shift == 0:
        return clip
    out = []
    for frame in clip:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
        out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    return out


def simulate_match_footage(clip: Clip, rng: np.random.Generator,
                           background: Optional[np.ndarray] = None) -> Clip:
    """Random in-match look for training: horizontal flip, hue shift,
    in-match character size on a 16:9 canvas, and stream compression. Expects
    frames that already went through strip_hitboxes."""
    if rng.random() < 0.5:
        clip = [np.ascontiguousarray(f[:, ::-1]) for f in clip]
    if rng.random() < 0.5:
        clip = shift_hue(clip, rng.uniform(-40, 40))
    clip = match_scale(clip, rng, height_frac=rng.uniform(0.35, 1.0), background=background)
    if rng.random() < 0.8:
        clip = degrade_stream(clip, short_side=int(rng.integers(140, CANVAS_H + 1)),
                              quality=int(rng.integers(20, 70)))
    return clip
