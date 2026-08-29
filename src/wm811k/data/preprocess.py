"""Convert the raw pickle into memory-mapped, fixed-size training arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from wm811k.data import CLASS_NAMES, CLASS_TO_INDEX
from wm811k.data.labels import normalize_group, normalize_label
from wm811k.data.split import create_grouped_splits
from wm811k.utils.config import load_config, require_keys


def resize_wafer_map(
    wafer_map: np.ndarray,
    image_size: int,
    preserve_aspect_ratio: bool = True,
) -> np.ndarray:
    if not isinstance(wafer_map, np.ndarray) or wafer_map.ndim != 2:
        raise ValueError("waferMap must be a two-dimensional NumPy array")
    discrete_map = np.asarray(wafer_map, dtype=np.float32)
    if preserve_aspect_ratio:
        height, width = discrete_map.shape
        side = max(height, width)
        pad_top = (side - height) // 2
        pad_bottom = side - height - pad_top
        pad_left = (side - width) // 2
        pad_right = side - width - pad_left
        discrete_map = np.pad(
            discrete_map,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
    tensor = torch.from_numpy(discrete_map)[None, None]
    resized = functional.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized[0, 0].to(torch.uint8).numpy()


def preprocess_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    require_keys(config, "raw_path", "processed_dir", "image_size", context="data config")
    raw_path = Path(config["raw_path"])
    processed_dir = Path(config["processed_dir"])
    image_size = int(config["image_size"])
    preserve_aspect_ratio = bool(config.get("preserve_aspect_ratio", True))
    seed = int(config.get("seed", 42))
    split_config = config.get("split", {})

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset does not exist: {raw_path}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    print(f"Loading raw dataset: {raw_path}")
    frame = pd.read_pickle(raw_path)
    required_columns = {"waferMap", "failureType"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise KeyError(f"Raw dataset is missing columns: {', '.join(missing)}")

    normalized_labels = frame["failureType"].map(normalize_label)
    valid_maps = frame["waferMap"].map(lambda value: isinstance(value, np.ndarray) and value.ndim == 2)
    selected = frame.loc[normalized_labels.notna() & valid_maps].copy()
    selected_labels = normalized_labels.loc[selected.index]
    sample_count = len(selected)
    if sample_count == 0:
        raise ValueError("No labeled wafer maps were found")

    processed_dir.mkdir(parents=True, exist_ok=True)
    images_path = processed_dir / "images.npy"
    images = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, image_size, image_size),
    )
    labels = np.empty(sample_count, dtype=np.int64)
    groups = []

    has_lot_name = "lotName" in selected.columns
    for output_index, (row_index, row) in enumerate(selected.iterrows()):
        images[output_index] = resize_wafer_map(
            row["waferMap"], image_size, preserve_aspect_ratio=preserve_aspect_ratio
        )
        labels[output_index] = CLASS_TO_INDEX[selected_labels.loc[row_index]]
        lot_value = row["lotName"] if has_lot_name else None
        groups.append(normalize_group(lot_value, fallback=f"row-{row_index}"))
        if (output_index + 1) % 10_000 == 0:
            print(f"Processed {output_index + 1:,}/{sample_count:,} wafer maps")

    images.flush()
    del images
    groups_array = np.asarray(groups, dtype=str)
    np.save(processed_dir / "labels.npy", labels)
    np.save(processed_dir / "groups.npy", groups_array)

    splits = create_grouped_splits(
        labels,
        groups_array,
        train_fraction=float(split_config.get("train", 0.70)),
        val_fraction=float(split_config.get("val", 0.15)),
        test_fraction=float(split_config.get("test", 0.15)),
        seed=seed,
        candidates=int(split_config.get("candidates", 64)),
    )
    np.savez_compressed(processed_dir / "splits.npz", **splits)

    class_counts = {
        CLASS_NAMES[index]: int((labels == index).sum()) for index in range(len(CLASS_NAMES))
    }
    summary = {
        "raw_rows": int(len(frame)),
        "processed_rows": int(sample_count),
        "image_size": image_size,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "input_encoding": "uint8 values 0=outside, 1=good die, 2=failed die",
        "class_names": list(CLASS_NAMES),
        "class_counts": class_counts,
        "unique_groups": int(len(np.unique(groups_array))),
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "seed": seed,
    }
    with (processed_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def preprocess_from_config(config_path: Path) -> Dict[str, Any]:
    return preprocess_dataset(load_config(config_path))
