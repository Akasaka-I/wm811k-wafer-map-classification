"""Standalone checkpoint evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from wm811k.data import CLASS_NAMES
from wm811k.data.dataset import build_dataloaders
from wm811k.evaluation.metrics import classification_metrics
from wm811k.evaluation.visualization import save_confusion_matrix
from wm811k.models.factory import build_model
from wm811k.utils.config import load_config
from wm811k.utils.device import resolve_device


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    split_name: str = "test",
) -> Dict[str, Any]:
    if split_name not in {"val", "test"}:
        raise ValueError(f"split_name must be 'val' or 'test', got {split_name!r}")
    config = load_config(config_path)
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))
    data_config = config["data"]
    loaders = build_dataloaders(
        Path(data_config["processed_dir"]),
        batch_size=int(data_config.get("batch_size", 128)),
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("model_config", config["model"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    targets: List[int] = []
    predictions: List[int] = []
    with torch.inference_mode():
        for images, labels in loaders[split_name]:
            logits = model(images.to(device, non_blocking=True))
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(targets, predictions, CLASS_NAMES)
    experiment_name = str(config["experiment"]["name"])
    output_dir = Path(config["experiment"].get("output_dir", "outputs/experiments"))
    output_dir = output_dir / experiment_name / "evaluation" / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    save_confusion_matrix(
        metrics["confusion_matrix"], CLASS_NAMES, output_dir / "confusion_matrix.png"
    )
    split_label = "Validation" if split_name == "val" else "Test"
    print(f"{split_label} macro-F1: {metrics['macro_f1']:.4f}")
    return metrics
