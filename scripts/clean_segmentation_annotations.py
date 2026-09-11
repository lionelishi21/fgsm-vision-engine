"""
Repairs data/ufd/segmentation_dataset/annotations.json.

The exported COCO file has two bugs stacked on top of each other:
  1. Each of the 26 real source frames appears 27-28x as separate "image"
     entries (fresh image_id, identical file_name) -> 715 declared images,
     which then leaks near-duplicate frames across the train/val/test split
     (a plain 80/10/10 index slice in train_hitbox_segmentation.py).
  2. Within a single image, each true labeled region (one per class) is
     stored as a few hundred to a few thousand near-identical jittered
     polygons instead of one clean polygon - almost certainly from tracing
     mask contours across many anti-aliased intensity levels without
     rounding/thresholding first.

This script collapses (1) by keeping one image entry per unique file_name,
and collapses (2) by rasterizing all near-duplicate polygons for a given
(image, category) into a single binary mask, unioning them, and re-extracting
clean contours from the union. A fixed-seed shuffle is applied to the
deduplicated image order so the downstream index-based split isn't
character-clustered (source frames are grouped ryu-then-ken).

Does not address the underlying data volume problem: 26 labeled frames is
very little for training and only 3 of 7 declared hitbox classes have any
annotations at all (hitbox, projectile_invincible, armor, mystery_box are
never labeled in this export).
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

MIN_CONTOUR_AREA = 20  # drop specks left over from the union, if any


def polygon_to_mask(segmentation, height, width):
    rles = mask_utils.frPyObjects(segmentation, height, width)
    m = mask_utils.decode(rles)
    if m.ndim == 3:
        m = m.max(axis=2)
    return m.astype(np.uint8)


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        if len(c) < 3:
            continue
        polygons.append(c.reshape(-1).astype(float).tolist())
    return polygons


def clean(input_path: Path, output_path: Path, seed: int = 42):
    with open(input_path) as f:
        coco = json.load(f)

    orig_images = coco["images"]
    orig_annotations = coco["annotations"]
    print(f"Loaded {len(orig_images)} image entries, {len(orig_annotations)} annotations")

    # --- Step 1: collapse duplicate image entries down to unique file_names ---
    first_seen = {}
    canonical_old_id_for_file = {}
    for img in orig_images:
        name = img["file_name"]
        if name not in first_seen:
            first_seen[name] = img
            canonical_old_id_for_file[name] = img["id"]

    unique_names = list(first_seen.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_names)

    new_id_for_file = {name: i for i, name in enumerate(unique_names)}
    new_images = []
    for name in unique_names:
        img = dict(first_seen[name])
        img["id"] = new_id_for_file[name]
        new_images.append(img)

    print(f"Collapsed to {len(new_images)} unique images")

    # --- Step 2: for each unique image, group its (already-duplicated-away)
    # annotations by category and merge near-duplicate polygons into one ---
    anns_by_old_image_id = defaultdict(list)
    for a in orig_annotations:
        anns_by_old_image_id[a["image_id"]].append(a)

    new_annotations = []
    next_ann_id = 0
    dims_by_old_id = {img["id"]: (img["height"], img["width"]) for img in orig_images}

    for name in unique_names:
        old_id = canonical_old_id_for_file[name]
        new_id = new_id_for_file[name]
        height, width = dims_by_old_id[old_id]

        anns = anns_by_old_image_id.get(old_id, [])
        by_category = defaultdict(list)
        for a in anns:
            by_category[a["category_id"]].append(a)

        for category_id, group in by_category.items():
            union_mask = np.zeros((height, width), dtype=np.uint8)
            for a in group:
                seg = a["segmentation"]
                if isinstance(seg, list):
                    m = polygon_to_mask(seg, height, width)
                    union_mask = np.maximum(union_mask, m)

            polygons = mask_to_polygons(union_mask)
            for poly in polygons:
                rle = mask_utils.frPyObjects([poly], height, width)
                area = float(mask_utils.area(rle[0]))
                bbox = mask_utils.toBbox(rle[0]).tolist()
                new_annotations.append({
                    "id": next_ann_id,
                    "image_id": new_id,
                    "category_id": category_id,
                    "segmentation": [poly],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                next_ann_id += 1

    print(f"Merged down to {len(new_annotations)} annotations")

    covered = sorted({a["category_id"] for a in new_annotations})
    all_cats = {c["id"]: c["name"] for c in coco["categories"]}
    missing = [name for cid, name in all_cats.items() if cid not in covered]
    print(f"Categories with data: {[all_cats[c] for c in covered]}")
    if missing:
        print(f"Categories with NO labeled examples (unchanged by this script): {missing}")

    coco["images"] = new_images
    coco["annotations"] = new_annotations

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"Wrote cleaned annotations to {output_path} "
          f"({output_path.stat().st_size / 1e6:.1f} MB, was {input_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ufd/segmentation_dataset/annotations.json")
    parser.add_argument("--in-place", action="store_true",
                         help="Back up the input to .bak and overwrite it with the cleaned version")
    parser.add_argument("--output", default="data/ufd/segmentation_dataset/annotations.cleaned.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.in_place:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)
        print(f"Backed up original to {backup_path}")
        output_path = input_path
    else:
        output_path = Path(args.output)

    clean(input_path, output_path)
