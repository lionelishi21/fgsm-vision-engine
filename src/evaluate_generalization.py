# src/evaluate_generalization.py
"""
Does the temporal move classifier generalize beyond its training clips?

The 93% figure from train_temporal_classifier.py can't answer that: the
dataset puts every clip into train, val and test alike, and there's exactly
one clip per move, so that number is training accuracy. This script measures
generalization two ways:

  probes  - Re-runs the training clips under one real-footage difference at a
            time (hitbox overlay removed, stream compression, match-scale
            camera, the 16-consecutive-frame windows api.py actually feeds,
            all combined) plus a pure-noise baseline. Not a held-out test -
            the clips are still ones the model trained on - but if accuracy
            collapses under a single change, the model learned a shortcut and
            won't generalize, whatever real footage later shows.

  real    - Scores labeled clips from real match footage. This is the actual
            generalization test. Manifest is a CSV with columns:
                video,start_sec,end_sec,character,move,tags
            `move` must match a training move name exactly ("Standing Light
            Punch"), or be "none" for a window where no trained move happens
            (measures false positives). `tags` is optional, ';'-separated
            (e.g. "360p;p2_side;source=evo_2026") and results are broken
            down by each tag. Hold out whole matches - never mix clips from
            one match into anything used for training.

Usage:
    python -m src.evaluate_generalization probes --model-dir models/temporal_move_classifier
    python -m src.evaluate_generalization real --manifest data/real_eval/manifest.csv
    # (copy configs/real_eval_manifest.example.csv; video paths resolve relative to the manifest)
    # pass --model-dir more than once to compare e.g. base vs fine-tuned
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.footage_sim import strip_hitboxes as _strip_hitboxes  # noqa: E402
from data.temporal_dataset import UFDTemporalDataset  # noqa: E402
from inference import FGSMInferencePipeline, NUM_FRAMES  # noqa: E402

Clip = List[np.ndarray]  # list of HxWx3 uint8 RGB frames


# ─── Probes ──────────────────────────────────────────────────────────────────

def _degrade_stream(clip: Clip, short_side: int = 120, quality: int = 30) -> Clip:
    """Fixed mid-severity version of UFDTemporalDataset._simulate_stream_quality
    (roughly a 360p stream, re-encoded), so results are comparable run to run."""
    out = []
    for frame in clip:
        H, W = frame.shape[:2]
        scale = short_side / min(H, W)
        small = cv2.resize(frame, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(back, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return out


def _match_scale(clip: Clip, rng: np.random.Generator) -> Clip:
    """Shrinks the character to roughly in-match size (about 40% of frame
    height) on a 16:9 canvas, the way the full broadcast frame reaches the
    model in api.py. Background is the clip's own heavily blurred first frame,
    so it isn't a trivially clean backdrop."""
    H, W = clip[0].shape[:2]
    canvas_h = int(H / 0.4)
    canvas_w = int(canvas_h * 16 / 9)
    bg = cv2.resize(cv2.GaussianBlur(clip[0], (0, 0), 25), (canvas_w, canvas_h))
    y = canvas_h - H - int(canvas_h * 0.05)
    x = int(rng.integers(0, canvas_w - W))
    out = []
    for frame in clip:
        c = bg.copy()
        c[y:y + H, x:x + W] = frame
        out.append(c)
    return out


def _consecutive_window(clip: Clip, rng: np.random.Generator) -> Clip:
    """What api.py actually does: 16 back-to-back frames from wherever the
    window happens to fall, not 16 frames spread over the whole move. Pads with
    the first/last frame (a stand-in for idle) so the window can straddle the
    move's start or end."""
    padded = [clip[0]] * (NUM_FRAMES // 2) + list(clip) + [clip[-1]] * (NUM_FRAMES // 2)
    start = int(rng.integers(0, len(padded) - NUM_FRAMES + 1))
    return padded[start:start + NUM_FRAMES]


def _noise(rng: np.random.Generator) -> Clip:
    return [rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]


PROBES: Dict[str, Callable[[Clip, np.random.Generator], Clip]] = {
    "clean (as trained)": lambda c, r: c,
    "hitbox overlay stripped": lambda c, r: [_strip_hitboxes(f) for f in c],
    "stream ~360p compressed": lambda c, r: _degrade_stream(c),
    "match-scale camera": _match_scale,
    "16 consecutive frames": _consecutive_window,
    "all combined": lambda c, r: _degrade_stream(
        _match_scale([_strip_hitboxes(f) for f in _consecutive_window(c, r)], r)),
}


# ─── Scoring ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _probs(pipeline: FGSMInferencePipeline, clip: Clip) -> np.ndarray:
    pixel_values = pipeline._preprocess(clip).to(pipeline.device)
    logits = pipeline.model(pixel_values=pixel_values).logits[0]
    return torch.softmax(logits, dim=-1).cpu().numpy()


def _ece(confs: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: how far stated confidence is from actual
    accuracy, averaged over confidence bins. 0 = confidence means what it says."""
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.any():
            err += m.mean() * abs(confs[m].mean() - correct[m].mean())
    return float(err)


def _summarize(rows: List[dict], threshold: float) -> dict:
    """rows: dicts with label_idx (None for 'none'), label_char, probs."""
    labeled = [r for r in rows if r["label_idx"] is not None]
    unlabeled = [r for r in rows if r["label_idx"] is None]
    out = {"n": len(rows)}
    if labeled:
        P = np.stack([r["probs"] for r in labeled])
        y = np.array([r["label_idx"] for r in labeled])
        top1 = P.argmax(1)
        conf = P.max(1)
        correct = (top1 == y).astype(float)
        top5 = (np.argsort(-P, axis=1)[:, :5] == y[:, None]).any(1)
        chars = np.array([r["label_char"] for r in labeled])
        pred_chars = np.array([r["idx_to_char"][i] for r, i in zip(labeled, top1)])
        kept = conf >= threshold
        out.update({
            "top1": correct.mean(),
            "top5": top5.mean(),
            "char_acc": (chars == pred_chars).mean(),
            "mean_conf": conf.mean(),
            "kept@thr": kept.mean(),
            "prec@thr": correct[kept].mean() if kept.any() else float("nan"),
            "ece": _ece(conf, correct),
        })
    if unlabeled:
        conf = np.stack([r["probs"] for r in unlabeled]).max(1)
        out["false_pos@thr"] = (conf >= threshold).mean()
        out["none_mean_conf"] = conf.mean()
    return out


def _print_table(title: str, results: Dict[str, dict], threshold: float):
    cols = ["n", "top1", "top5", "char_acc", "mean_conf", "kept@thr", "prec@thr", "ece", "false_pos@thr"]
    print(f"\n=== {title} (threshold {threshold}) ===")
    print(f"{'condition':34s}" + "".join(f"{c:>14s}" for c in cols))
    for name, r in results.items():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{'-':>14s}" if v is None else f"{v:>14d}" if c == "n" else f"{v:>14.3f}")
        print(f"{name[:34]:34s}" + "".join(cells))


def _chance_line(pipeline: FGSMInferencePipeline):
    n, c = len(pipeline.idx_to_class), len(pipeline.trained_characters)
    print(f"(chance: top1 {1/n:.3f}, top5 {5/n:.3f}, char_acc ~{1/c:.3f} with {c} characters)")


# ─── Modes ───────────────────────────────────────────────────────────────────

def run_probes(pipeline: FGSMInferencePipeline, data_dir: str, seed: int, n_noise: int) -> Dict[str, dict]:
    ds = UFDTemporalDataset(data_dir, split="test", augment=False)
    class_to_idx = {v: k for k, v in pipeline.idx_to_class.items()}
    idx_to_char = {i: c.split(":", 1)[0] for i, c in pipeline.idx_to_class.items()}
    clips = []
    for frame_paths, _, class_name in ds.samples:
        if class_name not in class_to_idx:
            continue  # model and local dataset disagree on this class; skip rather than mislabel
        frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frame_paths]
        clips.append((frames, class_to_idx[class_name], class_name.split(":", 1)[0]))
    print(f"Probing {len(clips)} training clips ({len(ds.samples) - len(clips)} skipped: not in model's label map)")

    results = {}
    for name, fn in PROBES.items():
        rng = np.random.default_rng(seed)  # same random choices for every model compared
        rows = [{"label_idx": y, "label_char": ch, "idx_to_char": idx_to_char, "probs": _probs(pipeline, fn(f, rng))}
                for f, y, ch in clips]
        results[name] = _summarize(rows, pipeline.confidence_threshold)
        print(f"  {name}: top1 {results[name]['top1']:.3f}", flush=True)

    rng = np.random.default_rng(seed)
    noise_rows = [{"label_idx": None, "probs": _probs(pipeline, _noise(rng))} for _ in range(n_noise)]
    results["pure noise (should be rejected)"] = _summarize(noise_rows, pipeline.confidence_threshold)
    return results


def _read_segment(video: Path, start: float, end: float) -> Clip:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    while cap.get(cv2.CAP_PROP_POS_MSEC) <= end * 1000:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def run_real(pipeline: FGSMInferencePipeline, manifest: Path) -> Dict[str, dict]:
    norm = lambda s: s.strip().lower()
    class_to_idx = {norm(v): k for k, v in pipeline.idx_to_class.items()}
    idx_to_char = {i: c.split(":", 1)[0] for i, c in pipeline.idx_to_class.items()}
    by_tag: Dict[str, List[dict]] = defaultdict(list)
    skipped = []

    with open(manifest, newline="") as f:
        for line_no, rec in enumerate(csv.DictReader(f), start=2):
            video = (manifest.parent / rec["video"]).resolve() if not Path(rec["video"]).is_absolute() else Path(rec["video"])
            move = rec["move"].strip()
            if norm(move) == "none":
                label_idx = None
            else:
                label_idx = class_to_idx.get(norm(f"{rec['character']}:{move}"))
                if label_idx is None:
                    skipped.append(f"line {line_no}: '{rec['character']}:{move}' isn't a trained class")
                    continue
            frames = _read_segment(video, float(rec["start_sec"]), float(rec["end_sec"]))
            if len(frames) < 2:
                skipped.append(f"line {line_no}: got {len(frames)} frame(s) from {video.name}")
                continue
            row = {"label_idx": label_idx, "label_char": norm(rec["character"]), "idx_to_char": idx_to_char,
                   "probs": _probs(pipeline, frames)}
            by_tag["all"].append(row)
            for tag in filter(None, (t.strip() for t in (rec.get("tags") or "").split(";"))):
                by_tag[f"tag:{tag}"].append(row)

    for s in skipped:
        print(f"  skipped {s}")
    if not by_tag:
        raise SystemExit("No usable rows in manifest.")
    return {tag: _summarize(rows, pipeline.confidence_threshold) for tag, rows in by_tag.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["probes", "real"])
    p.add_argument("--model-dir", action="append", help="repeatable, to compare models (default: models/temporal_move_classifier)")
    p.add_argument("--data-dir", default="data/ufd")
    p.add_argument("--manifest", type=Path, help="CSV for 'real' mode")
    p.add_argument("--threshold", type=float, default=0.12, help="same default as api.py's FGSM_CONFIDENCE_THRESHOLD")
    p.add_argument("--noise-clips", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    p.add_argument("--json-out", type=Path, help="also write results as JSON")
    args = p.parse_args()

    if args.mode == "real" and not args.manifest:
        p.error("real mode needs --manifest")
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    all_results = {}
    for model_dir in args.model_dir or ["models/temporal_move_classifier"]:
        pipeline = FGSMInferencePipeline(model_dir, device=device, confidence_threshold=args.threshold)
        print(f"\nModel: {model_dir} on {device} ({len(pipeline.idx_to_class)} classes)")
        results = run_probes(pipeline, args.data_dir, args.seed, args.noise_clips) if args.mode == "probes" \
            else run_real(pipeline, args.manifest)
        _print_table(f"{args.mode}: {model_dir}", results, args.threshold)
        _chance_line(pipeline)
        all_results[model_dir] = results

    if args.json_out:
        args.json_out.write_text(json.dumps(all_results, indent=2, default=float))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
