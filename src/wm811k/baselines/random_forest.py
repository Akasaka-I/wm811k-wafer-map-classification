"""Majority-class and Random Forest validation baselines."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from wm811k.baselines.features import FEATURE_NAMES, load_or_create_feature_cache
from wm811k.data import CLASS_NAMES
from wm811k.evaluation.metrics import classification_metrics
from wm811k.evaluation.visualization import save_confusion_matrix
from wm811k.utils.config import load_config, require_keys


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def train_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    require_keys(config, "experiment", "data", "features", "model", context="baseline config")
    experiment = config["experiment"]
    data_config = config["data"]
    feature_config = config["features"]
    model_config = config["model"]
    seed = int(experiment.get("seed", 42))
    name = str(experiment["name"])
    processed_dir = Path(data_config["processed_dir"])
    output_dir = Path(experiment.get("output_dir", "outputs/experiments")) / name
    checkpoint_dir = Path(experiment.get("checkpoint_dir", "outputs/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(processed_dir / "labels.npy", mmap_mode="r")
    splits = np.load(processed_dir / "splits.npz")
    train_indices = np.asarray(splits["train"])
    val_indices = np.asarray(splits["val"])
    features = load_or_create_feature_cache(
        processed_dir,
        Path(feature_config.get("cache_path", processed_dir / "spatial_features.npy")),
        batch_size=int(feature_config.get("batch_size", 2048)),
        force=bool(feature_config.get("force_rebuild", False)),
    )

    train_labels = np.asarray(labels[train_indices])
    val_labels = np.asarray(labels[val_indices])
    majority_class = int(np.bincount(train_labels, minlength=len(CLASS_NAMES)).argmax())
    majority_predictions = np.full_like(val_labels, majority_class)
    majority_metrics = classification_metrics(val_labels, majority_predictions, CLASS_NAMES)
    _write_json(output_dir / "majority_validation_metrics.json", majority_metrics)

    model = RandomForestClassifier(
        n_estimators=int(model_config.get("n_estimators", 300)),
        max_depth=model_config.get("max_depth", 24),
        min_samples_leaf=int(model_config.get("min_samples_leaf", 2)),
        max_features=model_config.get("max_features", "sqrt"),
        class_weight=model_config.get("class_weight", "balanced_subsample"),
        n_jobs=int(model_config.get("n_jobs", -1)),
        random_state=seed,
        verbose=int(model_config.get("verbose", 0)),
    )
    print(f"Training Random Forest on {len(train_indices):,} samples and {len(FEATURE_NAMES)} features")
    started = time.perf_counter()
    model.fit(np.asarray(features[train_indices]), train_labels)
    training_seconds = time.perf_counter() - started
    validation_predictions = model.predict(np.asarray(features[val_indices]))
    validation_metrics = classification_metrics(val_labels, validation_predictions, CLASS_NAMES)
    validation_metrics["training_seconds"] = training_seconds
    validation_metrics["feature_count"] = len(FEATURE_NAMES)
    _write_json(output_dir / "validation_metrics.json", validation_metrics)
    save_confusion_matrix(
        validation_metrics["confusion_matrix"],
        CLASS_NAMES,
        output_dir / "validation_confusion_matrix.png",
    )

    importances = sorted(
        (
            {"feature": feature_name, "importance": float(importance)}
            for feature_name, importance in zip(FEATURE_NAMES, model.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    _write_json(output_dir / "feature_importance.json", {"features": importances})
    checkpoint_path = checkpoint_dir / f"{name}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "class_names": CLASS_NAMES,
            "config": config,
        },
        checkpoint_path,
        compress=3,
    )
    print(f"Majority validation macro-F1: {majority_metrics['macro_f1']:.4f}")
    print(f"Random Forest validation macro-F1: {validation_metrics['macro_f1']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print("The test split was not evaluated.")
    return {
        "checkpoint": str(checkpoint_path),
        "majority_validation_metrics": majority_metrics,
        "validation_metrics": validation_metrics,
    }


def train_baseline_from_config(config_path: Path) -> Dict[str, Any]:
    return train_baseline(load_config(config_path))
