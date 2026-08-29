"""Generate the frozen model comparison, error analysis, and Grad-CAM report."""

from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from wm811k.analysis.gradcam import GradCAM
from wm811k.data import CLASS_NAMES
from wm811k.data.dataset import build_dataloaders
from wm811k.models.factory import build_model
from wm811k.utils.config import load_config
from wm811k.utils.device import resolve_device


WAFER_COLORS = ListedColormap(["#111827", "#d1d5db", "#ef4444"])


def _read_json(path: Path) -> Any:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _checkpoint_parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _collect_comparisons(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons = []
    for entry in entries:
        metrics = _read_json(Path(entry["validation_metrics"]))
        checkpoint_path = entry.get("checkpoint")
        comparisons.append(
            {
                "name": entry["name"],
                "kind": entry["kind"],
                "validation_macro_f1": float(metrics["macro_f1"]),
                "validation_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "validation_accuracy": float(metrics["accuracy"]),
                "parameters": (
                    _checkpoint_parameter_count(Path(checkpoint_path)) if checkpoint_path else None
                ),
                "history_path": entry.get("history"),
            }
        )
    return comparisons


def _save_comparison_files(comparisons: List[Dict[str, Any]], output_dir: Path) -> None:
    _write_json(output_dir / "model_comparison.json", comparisons)
    with (output_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    figure, axis = plt.subplots(figsize=(11, 6))
    ranked = sorted(comparisons, key=lambda entry: entry["validation_macro_f1"], reverse=True)
    names = [entry["name"] for entry in ranked]
    values = [entry["validation_macro_f1"] for entry in ranked]
    colors = ["#ef4444" if entry["kind"] == "failed" else "#3b82f6" for entry in ranked]
    bars = axis.barh(np.arange(len(names)), values, color=colors)
    axis.set_yticks(np.arange(len(names)), names)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Validation Macro F1")
    axis.set_title("Model selection on the validation split")
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "model_validation_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _save_training_curves(comparisons: List[Dict[str, Any]], output_dir: Path) -> None:
    figure, (f1_axis, resnet_axis) = plt.subplots(1, 2, figsize=(15, 5.8))
    for entry in comparisons:
        history_path = entry.get("history_path")
        if not history_path or entry["kind"] == "failed":
            continue
        history = _read_json(Path(history_path))
        f1_axis.plot(
            [record["epoch"] for record in history],
            [record["val_macro_f1"] for record in history],
            label=entry["name"],
            linewidth=1.8,
        )
        if entry["name"] == "ResNet18":
            epochs = [record["epoch"] for record in history]
            resnet_axis.plot(
                epochs,
                [record["train_macro_f1"] for record in history],
                label="Train Macro F1",
                linewidth=1.8,
            )
            resnet_axis.plot(
                epochs,
                [record["val_macro_f1"] for record in history],
                label="Validation Macro F1",
                linewidth=1.8,
            )
            best = max(history, key=lambda record: record["val_macro_f1"])
            resnet_axis.scatter(
                [best["epoch"]], [best["val_macro_f1"]], color="#ef4444", zorder=3
            )
            resnet_axis.annotate(
                f"Best: epoch {best['epoch']}\n{best['val_macro_f1']:.4f}",
                (best["epoch"], best["val_macro_f1"]),
                xytext=(8, -38),
                textcoords="offset points",
            )

    f1_axis.set_title("Validation performance during training")
    f1_axis.set_xlabel("Epoch")
    f1_axis.set_ylabel("Macro F1")
    f1_axis.set_ylim(0, 1)
    f1_axis.grid(alpha=0.25)
    f1_axis.legend()
    resnet_axis.set_title("Selected ResNet18 learning curve")
    resnet_axis.set_xlabel("Epoch")
    resnet_axis.set_ylabel("Macro F1")
    resnet_axis.set_ylim(0, 1)
    resnet_axis.grid(alpha=0.25)
    resnet_axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _save_test_class_metrics(test_metrics: Dict[str, Any], output_dir: Path) -> None:
    report = test_metrics["classification_report"]
    positions = np.arange(len(CLASS_NAMES))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 6))
    for offset, (metric, label, color) in enumerate(
        (
            ("precision", "Precision", "#3b82f6"),
            ("recall", "Recall", "#f59e0b"),
            ("f1-score", "F1", "#10b981"),
        )
    ):
        values = [report[class_name][metric] for class_name in CLASS_NAMES]
        axis.bar(positions + (offset - 1) * width, values, width, label=label, color=color)
    axis.set_ylim(0, 1.08)
    axis.set_xticks(positions, CLASS_NAMES, rotation=35, ha="right")
    axis.set_xlabel("Failure class")
    axis.set_ylabel("Score")
    axis.set_title("Final ResNet18 performance by test class")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "test_per_class_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _run_final_inference(
    config_path: Path,
    checkpoint_path: Path,
    processed_dir: Path,
) -> Tuple[torch.nn.Module, torch.device, Any, np.ndarray, np.ndarray, np.ndarray]:
    training_config = load_config(config_path)
    device = resolve_device(str(training_config.get("training", {}).get("device", "auto")))
    data_config = training_config["data"]
    loaders = build_dataloaders(
        processed_dir,
        batch_size=int(data_config.get("batch_size", 128)),
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loaders["test"]:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    probability_array = np.concatenate(probabilities)
    target_array = np.concatenate(targets)
    predictions = probability_array.argmax(axis=1)
    return model, device, loaders["test"].dataset, target_array, predictions, probability_array


def _error_records(
    dataset,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, Any]]:
    confidence = probabilities[np.arange(len(predictions)), predictions]
    error_positions = np.flatnonzero(targets != predictions)
    order = error_positions[np.argsort(confidence[error_positions])[::-1]]
    records = []
    for position in order:
        global_index = int(dataset.indices[position])
        records.append(
            {
                "dataset_position": int(position),
                "global_index": global_index,
                "lot": str(groups[global_index]),
                "true_class": CLASS_NAMES[int(targets[position])],
                "predicted_class": CLASS_NAMES[int(predictions[position])],
                "confidence": float(confidence[position]),
            }
        )
    return records


def _summarize_errors(
    records: Sequence[Dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    confidence = probabilities[np.arange(len(predictions)), predictions]
    correct = targets == predictions
    confusion_counts = Counter(
        (record["true_class"], record["predicted_class"]) for record in records
    )
    lot_counts = Counter(record["lot"] for record in records)
    return {
        "error_count": int((~correct).sum()),
        "error_rate": float((~correct).mean()),
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_error": float(confidence[~correct].mean()),
        "errors_at_or_above_0_9_confidence": int((confidence[~correct] >= 0.9).sum()),
        "top_confusions": [
            {"true_class": true, "predicted_class": predicted, "count": count}
            for (true, predicted), count in confusion_counts.most_common(12)
        ],
        "top_error_lots": [
            {"lot": lot, "error_count": count} for lot, count in lot_counts.most_common(10)
        ],
    }


def _save_example_grid(
    records: Sequence[Dict[str, Any]],
    images: np.ndarray,
    output_path: Path,
    title: str,
    maximum: int,
) -> None:
    selected = list(records[:maximum])
    if not selected:
        return
    columns = 5
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.1 * rows), squeeze=False)
    for axis, record in zip(axes.flat, selected):
        axis.imshow(images[record["global_index"]], cmap=WAFER_COLORS, vmin=0, vmax=2)
        axis.set_title(
            f"{record['true_class']} → {record['predicted_class']}\n"
            f"p={record['confidence']:.3f}, idx={record['global_index']}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_targeted_false_positives(
    records: Sequence[Dict[str, Any]],
    images: np.ndarray,
    output_path: Path,
    per_class: int,
) -> None:
    target_classes = ("Random", "Near-full")
    figure, axes = plt.subplots(
        len(target_classes), per_class, figsize=(3.0 * per_class, 6.2), squeeze=False
    )
    for row, class_name in enumerate(target_classes):
        selected = [record for record in records if record["predicted_class"] == class_name][
            :per_class
        ]
        for column in range(per_class):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column >= len(selected):
                axis.axis("off")
                continue
            record = selected[column]
            axis.imshow(images[record["global_index"]], cmap=WAFER_COLORS, vmin=0, vmax=2)
            axis.set_title(
                f"true={record['true_class']}\np={record['confidence']:.3f}", fontsize=9
            )
            if column == 0:
                axis.set_ylabel(f"Predicted\n{class_name}", rotation=0, ha="right", va="center")
    figure.suptitle("Highest-confidence false positives for the weakest precision classes")
    figure.tight_layout(rect=(0.05, 0, 1, 0.94))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _resolve_gradcam_layer(model: torch.nn.Module) -> torch.nn.Module:
    # At 64x64, layer4 is only 2x2. Layer3 retains a more useful 4x4 map.
    if hasattr(model, "layer3"):
        return model.layer3[-1]
    raise TypeError("The final report currently expects a ResNet-style model with layer3")


def _save_gradcam_grid(
    model: torch.nn.Module,
    device: torch.device,
    dataset,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> List[Dict[str, Any]]:
    figure, axes = plt.subplots(3, 6, figsize=(18, 10), squeeze=False)
    representatives = []
    confidence = probabilities[np.arange(len(predictions)), predictions]
    with GradCAM(model, _resolve_gradcam_layer(model)) as gradcam:
        for class_index, class_name in enumerate(CLASS_NAMES):
            candidates = np.flatnonzero((targets == class_index) & (predictions == class_index))
            if len(candidates) == 0:
                continue
            ordered = candidates[np.argsort(confidence[candidates])]
            position = int(ordered[len(ordered) // 2])
            global_index = int(dataset.indices[position])
            input_tensor, _ = dataset[position]
            activation_map, _ = gradcam.generate(input_tensor[None].to(device), class_index)
            wafer_map = np.asarray(dataset.images[global_index])
            cam = activation_map.cpu().numpy()
            cam = np.where(wafer_map > 0, cam, np.nan)
            row = class_index // 3
            pair_column = (class_index % 3) * 2
            original_axis = axes[row, pair_column]
            cam_axis = axes[row, pair_column + 1]
            original_axis.imshow(wafer_map, cmap=WAFER_COLORS, vmin=0, vmax=2)
            cam_axis.imshow(wafer_map, cmap="gray", vmin=0, vmax=2)
            cam_axis.imshow(cam, cmap="jet", alpha=0.52, vmin=0, vmax=1)
            original_axis.set_title(f"{class_name}\nwafer map")
            cam_axis.set_title(f"Grad-CAM\np={confidence[position]:.3f}")
            for axis in (original_axis, cam_axis):
                axis.set_xticks([])
                axis.set_yticks([])
            representatives.append(
                {
                    "class": class_name,
                    "global_index": global_index,
                    "confidence": float(confidence[position]),
                }
            )
    figure.suptitle("ResNet18 Grad-CAM on correctly classified test examples")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return representatives


def _save_markdown_report(
    path: Path,
    comparisons: List[Dict[str, Any]],
    test_metrics: Dict[str, Any],
    error_count: int,
    test_size: int,
    error_summary: Dict[str, Any],
) -> None:
    validation_rows = "\n".join(
        f"| {entry['name']} | {entry['validation_macro_f1']:.4f} | "
        f"{entry['validation_balanced_accuracy']:.4f} | "
        f"{entry['parameters']:,} |"
        if entry["parameters"] is not None
        else f"| {entry['name']} | {entry['validation_macro_f1']:.4f} | "
        f"{entry['validation_balanced_accuracy']:.4f} | N/A |"
        for entry in sorted(comparisons, key=lambda item: item["validation_macro_f1"], reverse=True)
    )
    class_rows = "\n".join(
        f"| {class_name} | {test_metrics['classification_report'][class_name]['precision']:.3f} | "
        f"{test_metrics['classification_report'][class_name]['recall']:.3f} | "
        f"{test_metrics['classification_report'][class_name]['f1-score']:.3f} | "
        f"{int(test_metrics['classification_report'][class_name]['support'])} |"
        for class_name in CLASS_NAMES
    )
    confusion_rows = "\n".join(
        f"| {item['true_class']} | {item['predicted_class']} | {item['count']} |"
        for item in error_summary["top_confusions"][:8]
    )
    content = f"""# WM-811K Final Experiment Report

## Experimental protocol

The project uses a fixed 70/15/15 split grouped by `lotName`, preventing wafers from the same
production lot from appearing in multiple splits. Model selection used validation Macro F1 only.
After selecting ResNet18, the test split was evaluated once and then frozen.

## Model selection

| Model | Validation Macro F1 | Validation balanced accuracy | Parameters |
|---|---:|---:|---:|
{validation_rows}

ResNet18 was selected with validation Macro F1 `0.8740`. The original Swin-Tiny run is retained as
an optimization-collapse case study; its stabilized BF16 run reached `0.8633`.

![Validation comparison](outputs/final_report/model_validation_comparison.png)

![Training curves](outputs/final_report/training_curves.png)

## Frozen test result

- Macro F1: **{test_metrics['macro_f1']:.4f}**
- Balanced accuracy: **{test_metrics['balanced_accuracy']:.4f}**
- Accuracy: **{test_metrics['accuracy']:.4f}**
- Weighted F1: **{test_metrics['weighted_f1']:.4f}**
- Errors: **{error_count:,} / {test_size:,}**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{class_rows}

![Per-class test metrics](outputs/final_report/test_per_class_metrics.png)

![Test confusion matrix](outputs/experiments/resnet18_64/evaluation/test/confusion_matrix.png)

## Error analysis

The main Macro-F1 reduction from validation (`0.8740`) to test (`{test_metrics['macro_f1']:.4f}`)
comes from precision on the small Random and Near-full classes. Their recall remains high, but
class-weighted training produces false positives. Scratch generalizes better on test than on
validation, while common spatial patterns such as Edge-Ring remain stable.

- Mean confidence on errors: `{error_summary['mean_confidence_error']:.3f}`
- Errors with confidence >= 0.9: `{error_summary['errors_at_or_above_0_9_confidence']}`
- Lot with most errors: `{error_summary['top_error_lots'][0]['lot']}`
  (`{error_summary['top_error_lots'][0]['error_count']}` errors)

| True class | Predicted class | Count |
|---|---|---:|
{confusion_rows}

![High-confidence errors](outputs/final_report/high_confidence_errors.png)

![Targeted false positives](outputs/final_report/random_near_full_false_positives.png)

## Explainability

Grad-CAM is applied after model selection for descriptive analysis only. It should highlight the
spatial regions used by ResNet18, but it is not a causal explanation and was not used to tune the
model.

![Grad-CAM examples](outputs/final_report/gradcam_correct_examples.png)

## Limitations

- WM-811K is severely imbalanced; Near-full has only 25 test examples.
- `none` means no assigned failure pattern, not necessarily zero failed dies.
- Maps are padded and resized to 64x64, which can remove fine die-level detail.
- Results come from one fixed lot-aware split and have no external-fab validation.
- The test set has been consumed and must not be used for further model selection.
"""
    path.write_text(content, encoding="utf-8")


def build_final_report(config: Dict[str, Any]) -> Dict[str, Any]:
    final_config = config["final_model"]
    output_config = config.get("output", {})
    output_dir = Path(output_config.get("directory", "outputs/final_report"))
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = _collect_comparisons(config["comparisons"])
    _save_comparison_files(comparisons, output_dir)
    _save_training_curves(comparisons, output_dir)

    test_metrics = _read_json(Path(config["test_metrics"]))
    _save_test_class_metrics(test_metrics, output_dir)
    model, device, dataset, targets, predictions, probabilities = _run_final_inference(
        Path(final_config["config_path"]),
        Path(final_config["checkpoint_path"]),
        Path(final_config["processed_dir"]),
    )
    processed_dir = Path(final_config["processed_dir"])
    groups = np.load(processed_dir / "groups.npy", mmap_mode="r")
    images = np.load(processed_dir / "images.npy", mmap_mode="r")
    records = _error_records(dataset, targets, predictions, probabilities, groups)
    _write_json(output_dir / "high_confidence_errors.json", records)
    error_summary = _summarize_errors(records, targets, predictions, probabilities)
    _write_json(output_dir / "error_summary.json", error_summary)
    _save_example_grid(
        records,
        images,
        output_dir / "high_confidence_errors.png",
        "Highest-confidence ResNet18 test errors",
        int(output_config.get("error_examples", 20)),
    )
    _save_targeted_false_positives(
        records,
        images,
        output_dir / "random_near_full_false_positives.png",
        int(output_config.get("targeted_false_positives_per_class", 6)),
    )
    gradcam_representatives = _save_gradcam_grid(
        model,
        device,
        dataset,
        targets,
        predictions,
        probabilities,
        output_dir / "gradcam_correct_examples.png",
    )
    _write_json(output_dir / "gradcam_examples.json", gradcam_representatives)

    summary = {
        "selected_model": final_config["name"],
        "validation_macro_f1": next(
            entry["validation_macro_f1"]
            for entry in comparisons
            if entry["name"] == final_config["name"]
        ),
        "test_metrics": {
            key: test_metrics[key]
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "test_samples": int(len(targets)),
        "test_errors": int((targets != predictions).sum()),
        "report_outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    _write_json(output_dir / "summary.json", summary)
    _save_markdown_report(
        Path(output_config.get("markdown_path", "FINAL_REPORT.md")),
        comparisons,
        test_metrics,
        summary["test_errors"],
        summary["test_samples"],
        error_summary,
    )
    print(f"Final report: {output_config.get('markdown_path', 'FINAL_REPORT.md')}")
    print(f"Artifacts: {output_dir}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    return summary


def build_final_report_from_config(config_path: Path) -> Dict[str, Any]:
    return build_final_report(load_config(config_path))
