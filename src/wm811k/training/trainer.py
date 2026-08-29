"""Model-agnostic PyTorch training loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wm811k.data import CLASS_NAMES
from wm811k.data.dataset import build_dataloaders
from wm811k.evaluation.metrics import classification_metrics
from wm811k.models.factory import build_model
from wm811k.training.losses import balanced_class_weights, build_loss
from wm811k.training.optimizer import build_optimizer
from wm811k.training.scheduler import build_scheduler
from wm811k.utils.config import load_config, require_keys
from wm811k.utils.device import resolve_device
from wm811k.utils.seed import seed_everything


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
    mixed_precision: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    gradient_clip_norm: float = 0.0,
) -> Tuple[float, List[int], List[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    processed_samples = 0
    targets: List[int] = []
    predictions: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach()) * len(labels)
        processed_samples += len(labels)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / max(processed_samples, 1), targets, predictions


def train_from_config(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    require_keys(config, "experiment", "data", "model", "training", context="training config")
    experiment = config["experiment"]
    data_config = config["data"]
    training_config = config["training"]
    name = str(experiment["name"])
    seed = int(experiment.get("seed", 42))
    epochs = int(training_config.get("epochs", 50))
    patience = int(training_config.get("early_stopping_patience", 8))
    early_stopping_min_epochs = int(training_config.get("early_stopping_min_epochs", 1))
    gradient_clip_norm = float(training_config.get("gradient_clip_norm", 0.0))
    device = resolve_device(str(training_config.get("device", "auto")))
    mixed_precision = bool(training_config.get("mixed_precision", True))
    amp_dtype_name = str(training_config.get("amp_dtype", "float16")).lower()
    amp_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if amp_dtype_name not in amp_dtypes:
        raise ValueError("training.amp_dtype must be 'float16' or 'bfloat16'")
    amp_dtype = amp_dtypes[amp_dtype_name]
    seed_everything(seed)

    loaders = build_dataloaders(
        Path(data_config["processed_dir"]),
        batch_size=int(data_config.get("batch_size", 128)),
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config["model"]).to(device)
    train_dataset = loaders["train"].dataset
    train_labels = np.asarray(train_dataset.labels)[train_dataset.indices]
    loss_config = config.get("loss", {})
    weights = balanced_class_weights(
        train_labels,
        len(CLASS_NAMES),
        power=float(loss_config.get("class_weight_power", 1.0)),
        max_weight=(
            float(loss_config["max_class_weight"])
            if loss_config.get("max_class_weight") is not None
            else None
        ),
    ).to(device)
    criterion = build_loss(loss_config, class_weights=weights).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}), epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=mixed_precision and device.type == "cuda" and amp_dtype == torch.float16,
    )

    checkpoint_dir = Path(experiment.get("checkpoint_dir", "outputs/checkpoints"))
    output_dir = Path(experiment.get("output_dir", "outputs/experiments")) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_best.pt"
    history: List[Dict[str, float]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    print(f"Training {config['model']['name']} on {device} for up to {epochs} epochs")
    print(
        "Class weights: "
        + ", ".join(
            f"{class_name}={weight:.3f}"
            for class_name, weight in zip(CLASS_NAMES, weights.detach().cpu().tolist())
        )
    )
    for epoch in range(1, epochs + 1):
        train_loss, train_targets, train_predictions = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            amp_dtype=amp_dtype,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_loss, val_targets, val_predictions = _run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            # Keep model selection reproducible across standalone validation runs.
            mixed_precision=False,
            amp_dtype=amp_dtype,
        )
        train_metrics = classification_metrics(train_targets, train_predictions, CLASS_NAMES)
        val_metrics = classification_metrics(val_targets, val_predictions, CLASS_NAMES)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val macro-F1 {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "class_names": list(CLASS_NAMES),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if epoch >= early_stopping_min_epochs and epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print("The test split was not evaluated. Use the evaluate command only after model selection.")
    return {
        "checkpoint": str(checkpoint_path),
        "best_validation_macro_f1": best_macro_f1,
    }
