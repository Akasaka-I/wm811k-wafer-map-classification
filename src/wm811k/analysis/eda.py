"""Train-only exploratory data analysis for WM-811K."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from wm811k.data import CLASS_NAMES
from wm811k.utils.config import load_config


def _load_processed(processed_dir: Path):
    required = {
        "images": processed_dir / "images.npy",
        "labels": processed_dir / "labels.npy",
        "groups": processed_dir / "groups.npy",
        "splits": processed_dir / "splits.npz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed data files:\n"
            + "\n".join(missing)
            + "\nRun `python main.py preprocess --config configs/data.yaml` first."
        )
    return (
        np.load(required["images"], mmap_mode="r"),
        np.load(required["labels"], mmap_mode="r"),
        np.load(required["groups"], mmap_mode="r"),
        np.load(required["splits"]),
    )


def _chunked(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _compute_map_statistics(
    images: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    defect_ratios = np.empty(len(indices), dtype=np.float32)
    valid_die_counts = np.empty(len(indices), dtype=np.int32)
    offset = 0
    for batch_indices in _chunked(indices, batch_size):
        batch = np.asarray(images[batch_indices])
        valid = (batch > 0).sum(axis=(1, 2))
        failed = (batch == 2).sum(axis=(1, 2))
        count = len(batch_indices)
        valid_die_counts[offset : offset + count] = valid
        defect_ratios[offset : offset + count] = np.divide(
            failed,
            valid,
            out=np.zeros_like(failed, dtype=np.float64),
            where=valid > 0,
        )
        offset += count
    return defect_ratios, valid_die_counts


def _save_class_distribution(
    labels: np.ndarray,
    splits,
    output_path: Path,
) -> Dict[str, Dict[str, int]]:
    split_names = ("train", "val", "test")
    counts_by_split: Dict[str, Dict[str, int]] = {}
    train_counts = None
    for split_name in split_names:
        counts = np.bincount(labels[splits[split_name]], minlength=len(CLASS_NAMES))
        counts_by_split[split_name] = {
            class_name: int(counts[index]) for index, class_name in enumerate(CLASS_NAMES)
        }
        if split_name == "train":
            train_counts = counts

    figure, axis = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(CLASS_NAMES))
    bars = axis.bar(positions, train_counts, color="#3b82f6")
    axis.set_yscale("log")
    axis.set_title("WM-811K training-class distribution")
    axis.set_xlabel("Failure class")
    axis.set_ylabel("Number of wafer maps (log scale)")
    axis.set_xticks(positions, CLASS_NAMES, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, train_counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return counts_by_split


def _save_split_proportions(labels: np.ndarray, splits, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(CLASS_NAMES))
    width = 0.25
    colors = ("#2563eb", "#f59e0b", "#10b981")
    for offset, (split_name, color) in enumerate(zip(("train", "val", "test"), colors)):
        counts = np.bincount(labels[splits[split_name]], minlength=len(CLASS_NAMES))
        proportions = counts / counts.sum() * 100
        axis.bar(
            positions + (offset - 1) * width,
            proportions,
            width,
            label=split_name,
            color=color,
        )
    axis.set_yscale("log")
    axis.set_title("Class proportions by split")
    axis.set_xlabel("Failure class")
    axis.set_ylabel("Share of split (%) — log scale")
    axis.set_xticks(positions, CLASS_NAMES, rotation=35, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _save_sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    output_path: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    random = np.random.default_rng(seed)
    figure, axes = plt.subplots(
        len(CLASS_NAMES),
        samples_per_class,
        figsize=(2.35 * samples_per_class, 2.05 * len(CLASS_NAMES)),
        squeeze=False,
    )
    wafer_colors = ListedColormap(["#111827", "#d1d5db", "#ef4444"])
    train_labels = labels[train_indices]
    for class_index, class_name in enumerate(CLASS_NAMES):
        candidates = train_indices[train_labels == class_index]
        selected = random.choice(
            candidates,
            size=min(samples_per_class, len(candidates)),
            replace=False,
        )
        for column in range(samples_per_class):
            axis = axes[class_index, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column < len(selected):
                axis.imshow(images[int(selected[column])], cmap=wafer_colors, vmin=0, vmax=2)
            else:
                axis.axis("off")
            if column == 0:
                axis.set_ylabel(class_name, rotation=0, ha="right", va="center", labelpad=35)
    figure.suptitle("Training examples: gray = good die, red = failed die", y=0.995)
    figure.tight_layout(rect=(0.08, 0, 1, 0.985))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_defect_ratio_distribution(
    labels: np.ndarray,
    train_indices: np.ndarray,
    defect_ratios: np.ndarray,
    output_path: Path,
) -> Dict[str, Dict[str, float]]:
    train_labels = labels[train_indices]
    values_by_class = [defect_ratios[train_labels == index] * 100 for index in range(len(CLASS_NAMES))]
    figure, axis = plt.subplots(figsize=(12, 6))
    boxplot = axis.boxplot(values_by_class, labels=CLASS_NAMES, showfliers=False, patch_artist=True)
    for box in boxplot["boxes"]:
        box.set_facecolor("#60a5fa")
        box.set_alpha(0.65)
    axis.set_title("Failed-die ratio by training class")
    axis.set_xlabel("Failure class")
    axis.set_ylabel("Failed dies among valid dies (%)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    statistics: Dict[str, Dict[str, float]] = {}
    for class_name, values in zip(CLASS_NAMES, values_by_class):
        statistics[class_name] = {
            "median_percent": float(np.median(values)),
            "q1_percent": float(np.quantile(values, 0.25)),
            "q3_percent": float(np.quantile(values, 0.75)),
        }
    return statistics


def _save_lot_distribution(groups: np.ndarray, train_indices: np.ndarray, output_path: Path) -> Dict[str, float]:
    _, lot_sizes = np.unique(groups[train_indices], return_counts=True)
    upper = max(int(lot_sizes.max()), 2)
    bins = np.unique(np.geomspace(1, upper + 1, num=30).astype(int))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.hist(lot_sizes, bins=bins, color="#8b5cf6", edgecolor="#ffffff", linewidth=0.5)
    axis.set_xscale("log")
    axis.set_title("Training wafers per lot")
    axis.set_xlabel("Number of labeled wafers in a lot (log scale)")
    axis.set_ylabel("Number of lots")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return {
        "unique_train_lots": int(len(lot_sizes)),
        "median_wafers_per_lot": float(np.median(lot_sizes)),
        "max_wafers_per_lot": int(lot_sizes.max()),
    }


def run_eda(config: Dict[str, Any]) -> Dict[str, Any]:
    processed_dir = Path(config["processed_dir"])
    eda_config = config.get("eda", {})
    output_dir = Path(eda_config.get("output_dir", "outputs/figures/eda"))
    summary_path = Path(eda_config.get("summary_path", "outputs/experiments/eda_summary.json"))
    samples_per_class = int(eda_config.get("samples_per_class", 4))
    batch_size = int(eda_config.get("analysis_batch_size", 4096))
    seed = int(config.get("seed", 42))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels, groups, splits = _load_processed(processed_dir)
    train_indices = np.asarray(splits["train"])
    defect_ratios, valid_die_counts = _compute_map_statistics(images, train_indices, batch_size)

    counts_by_split = _save_class_distribution(
        labels, splits, output_dir / "class_distribution_train.png"
    )
    _save_split_proportions(labels, splits, output_dir / "class_proportions_by_split.png")
    _save_sample_grid(
        images,
        labels,
        train_indices,
        output_dir / "class_examples_train.png",
        samples_per_class,
        seed,
    )
    defect_statistics = _save_defect_ratio_distribution(
        labels,
        train_indices,
        defect_ratios,
        output_dir / "defect_ratio_by_class.png",
    )
    lot_statistics = _save_lot_distribution(
        groups, train_indices, output_dir / "train_lot_size_distribution.png"
    )

    train_counts = np.bincount(labels[train_indices], minlength=len(CLASS_NAMES))
    positive_counts = train_counts[train_counts > 0]
    summary = {
        "analysis_scope": "training split for image-derived statistics; all splits for count comparison",
        "processed_dir": str(processed_dir.resolve()),
        "image_shape": list(images.shape[1:]),
        "split_sizes": {name: int(len(splits[name])) for name in ("train", "val", "test")},
        "class_counts_by_split": counts_by_split,
        "train_imbalance_ratio_largest_to_smallest": float(positive_counts.max() / positive_counts.min()),
        "valid_die_count": {
            "median": float(np.median(valid_die_counts)),
            "min": int(valid_die_counts.min()),
            "max": int(valid_die_counts.max()),
        },
        "defect_ratio_by_class": defect_statistics,
        "lot_statistics": lot_statistics,
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"EDA summary: {summary_path}")
    print(f"Figures: {output_dir}")
    for figure_name in summary["figures"]:
        print(f"  - {figure_name}")
    return summary


def run_eda_from_config(config_path: Path) -> Dict[str, Any]:
    return run_eda(load_config(config_path))
