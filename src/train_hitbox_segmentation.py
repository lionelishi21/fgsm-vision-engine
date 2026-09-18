# src/train_hitbox_segmentation.py
"""
Train a segmentation model on UFD color-coded hitbox data.
Uses Mask2Former or U-Net to segment hitboxes/hurtboxes from character crops.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from transformers import TrainingArguments, Trainer


class UFDSegmentationDataset(Dataset):
    def __init__(self, data_dir: str, processor, split="train"):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        
        with open(self.data_dir / "annotations.json") as f:
            self.coco = json.load(f)
        
        # Split 80/10/10
        all_images = self.coco["images"]
        n = len(all_images)
        if split == "train":
            self.images = all_images[:int(n*0.8)]
        elif split == "val":
            self.images = all_images[int(n*0.8):int(n*0.9)]
        else:
            self.images = all_images[int(n*0.9):]

        self.id_to_img = {img["id"]: img for img in self.coco["images"]}
        self.img_to_anns = {}
        for ann in self.coco["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.data_dir / "images" / img_info["file_name"]
        mask_path = self.data_dir / "masks" / img_info["file_name"].replace(".png", "_mask.png")

        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        # Build instance map and class labels
        instances = []
        class_labels = []
        masks_list = []

        # Group annotations by class
        class_masks = {}
        for ann in self.img_to_anns.get(img_info["id"], []):
            class_id = ann["category_id"]
            if class_id not in class_masks:
                class_masks[class_id] = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
            
            # Rasterize segmentation
            from pycocotools import mask as mask_utils
            if isinstance(ann["segmentation"], list):
                # Polygon
                rles = mask_utils.frPyObjects(ann["segmentation"], img_info["height"], img_info["width"])
                m = mask_utils.decode(rles)
                if m.ndim == 3:
                    m = m.max(axis=2)
                class_masks[class_id] = np.maximum(class_masks[class_id], m)

        # Convert to instance format
        instance_id = 0
        semantic_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.int64)
        
        for class_id, binary_mask in class_masks.items():
            if binary_mask.sum() == 0:
                continue
            
            # Label connected components as separate instances
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary_mask)
            
            for i in range(1, num_features + 1):
                instance_mask = (labeled == i).astype(np.float32)
                instances.append(instance_mask)
                class_labels.append(class_id)
                semantic_mask[labeled == i] = class_id

        if len(instances) == 0:
            # Empty image - add dummy
            instances = [np.zeros((img_info["height"], img_info["width"]), dtype=np.float32)]
            class_labels = [0]

        # Prepare inputs for Mask2Former
        inputs = self.processor(
            images=[image],
            segmentation_maps=[semantic_mask],
            instance_id_to_semantic_id={i: cls for i, cls in enumerate(class_labels)},
            return_tensors="pt",
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        return inputs


def find_last_checkpoint(output_dir: str, expected_train_size: int):
    # output_dir is often a persistent path (e.g. a mounted Google Drive
    # folder) reused across unrelated experiments. Blindly resuming from
    # whatever checkpoint happens to be there once silently continued
    # training from a completely different, incompatible dataset - so
    # a fingerprint of the current dataset size must match before trusting
    # any existing checkpoint.
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        return None

    fingerprint_path = out / "dataset_fingerprint.json"
    recorded_size = None
    if fingerprint_path.exists():
        try:
            recorded_size = json.loads(fingerprint_path.read_text()).get("num_train_images")
        except (json.JSONDecodeError, OSError):
            pass

    if recorded_size != expected_train_size:
        print(
            f"Ignoring {len(checkpoints)} existing checkpoint(s) in {output_dir}: "
            f"recorded dataset size ({recorded_size}) doesn't match the current "
            f"dataset ({expected_train_size} images) - starting fresh instead of "
            f"resuming from a possibly incompatible run."
        )
        return None
    return str(checkpoints[-1])


def train(data_dir: str, output_dir: str):
    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance"
    )
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance",
        num_labels=8,  # 0=background + 7 hitbox classes
        ignore_mismatched_sizes=True,
    )

    train_dataset = UFDSegmentationDataset(data_dir, processor, "train")
    val_dataset = UFDSegmentationDataset(data_dir, processor, "val")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=50,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        # Step-based, not epoch-based: on the full dataset one epoch is
        # ~980 steps, so a capped/interrupted run (Colab session limits,
        # a max_steps override, etc.) could otherwise finish without ever
        # crossing a single epoch boundary - and therefore without ever
        # saving a checkpoint at all.
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    resume_from = find_last_checkpoint(output_dir, expected_train_size=len(train_dataset))
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "dataset_fingerprint.json").write_text(
        json.dumps({"num_train_images": len(train_dataset)})
    )

    trainer.train(resume_from_checkpoint=resume_from)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ufd/segmentation_dataset")
    parser.add_argument("--output", default="runs/hitbox_segmenter")
    args = parser.parse_args()
    train(args.data, args.output)
