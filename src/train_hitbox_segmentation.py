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
        eval_strategy="epoch",
        save_strategy="epoch",
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

    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ufd/segmentation_dataset")
    parser.add_argument("--output", default="runs/hitbox_segmenter")
    args = parser.parse_args()
    train(args.data, args.output)
