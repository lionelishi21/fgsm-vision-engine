# src/train_temporal_classifier.py
"""
Train Video Swin Transformer for temporal move recognition.
Input: 16-frame clip → Output: Move class (e.g., 'ryu:hadoken')
"""
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report

from data.temporal_dataset import UFDTemporalDataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Top-1 and Top-5 accuracy
    top1 = accuracy_score(labels, preds)
    
    metrics = {
        "accuracy": float(top1),
    }
    
    # Top-5 if we have enough classes
    if predictions.shape[1] >= 5:
        top5 = top_k_accuracy_score(labels, predictions, k=5)
        metrics["top5_accuracy"] = float(top5)
    
    return metrics


class FGSMVideoTrainer(Trainer):
    """Custom trainer with mixup/cutmix for video data."""
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def find_last_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(checkpoints[-1]) if checkpoints else None


def train(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config.get("data_dir", "data/ufd"))
    output_dir = Path(config.get("output_dir", "runs/temporal_move_classifier"))
    
    # Load datasets
    train_dataset = UFDTemporalDataset(str(data_dir), split="train", num_frames=16)
    val_dataset = UFDTemporalDataset(str(data_dir), split="val", num_frames=16)
    
    num_classes = len(train_dataset.class_to_idx)
    print(f"Training on {num_classes} move classes")
    
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({
            "class_to_idx": train_dataset.class_to_idx,
            "idx_to_class": train_dataset.idx_to_class,
        }, f, indent=2)

    model_name = config.get("model", "microsoft/video-swin-tiny-kinetics-400")
    
    try:
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    except Exception:
        print("Loading from config instead of pretrained...")
        model_config = VideoMAEConfig.from_pretrained(model_name)
        model_config.num_labels = num_classes
        model = VideoMAEForVideoClassification(model_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("epochs", 50),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=float(config.get("lr", 1e-4)),
        weight_decay=config.get("weight_decay", 0.05),
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = FGSMVideoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get("early_stopping_patience", 7))],
        mixup_alpha=config.get("mixup_alpha", 0.2),
    )

    resume_from = find_last_checkpoint(output_dir)
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(output_dir)
    
    test_dataset = UFDTemporalDataset(str(data_dir), split="test", num_frames=16)
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Accuracy: {test_results['eval_accuracy']:.4f}")
    
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    print("\n=== Per-Class Accuracy ===")
    char_correct = {}
    char_total = {}
    for pred, label in zip(preds, labels):
        class_name = test_dataset.idx_to_class[label]
        character = class_name.split(":")[0]
        char_total[character] = char_total.get(character, 0) + 1
        if pred == label:
            char_correct[character] = char_correct.get(character, 0) + 1
    
    for char in sorted(char_total.keys()):
        acc = char_correct.get(char, 0) / char_total[char]
        print(f"  {char:12s}: {acc:.2%} ({char_correct.get(char, 0)}/{char_total[char]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/temporal_classifier.yaml")
    args = parser.parse_args()
    train(args.config)
