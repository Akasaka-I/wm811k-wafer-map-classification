"""Spatial feature extraction from discrete wafer maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


BASE_FEATURE_NAMES = [
    "valid_area_fraction",
    "defect_ratio",
    "defect_center_x",
    "defect_center_y",
    "defect_spread_x",
    "defect_spread_y",
    "defect_covariance_xy",
]
RADIAL_FEATURE_NAMES = [f"radial_density_{index}" for index in range(5)]
ANGULAR_FEATURE_NAMES = [f"angular_density_{index}" for index in range(8)]
GRID_FEATURE_NAMES = [f"grid_density_r{row}_c{column}" for row in range(4) for column in range(4)]
ROW_FEATURE_NAMES = [f"row_band_density_{index}" for index in range(8)]
COLUMN_FEATURE_NAMES = [f"column_band_density_{index}" for index in range(8)]
FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + RADIAL_FEATURE_NAMES
    + ANGULAR_FEATURE_NAMES
    + GRID_FEATURE_NAMES
    + ROW_FEATURE_NAMES
    + COLUMN_FEATURE_NAMES
)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )


def _geometry_masks(height: int, width: int) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    y_coordinates = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_coordinates = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    radius = np.sqrt(x_grid**2 + y_grid**2)
    angle = np.arctan2(y_grid, x_grid)
    masks: List[np.ndarray] = []

    radial_edges = (0.0, 0.2, 0.4, 0.6, 0.8, np.inf)
    for lower, upper in zip(radial_edges[:-1], radial_edges[1:]):
        masks.append((radius >= lower) & (radius < upper))

    angular_edges = np.linspace(-np.pi, np.pi, 9)
    for lower, upper in zip(angular_edges[:-1], angular_edges[1:]):
        masks.append((angle >= lower) & (angle < upper))

    row_slices = np.array_split(np.arange(height), 4)
    column_slices = np.array_split(np.arange(width), 4)
    for rows in row_slices:
        for columns in column_slices:
            mask = np.zeros((height, width), dtype=bool)
            mask[np.ix_(rows, columns)] = True
            masks.append(mask)

    for rows in np.array_split(np.arange(height), 8):
        mask = np.zeros((height, width), dtype=bool)
        mask[rows, :] = True
        masks.append(mask)

    for columns in np.array_split(np.arange(width), 8):
        mask = np.zeros((height, width), dtype=bool)
        mask[:, columns] = True
        masks.append(mask)
    return x_grid, y_grid, masks


def extract_spatial_features(maps: np.ndarray) -> np.ndarray:
    maps = np.asarray(maps)
    if maps.ndim != 3:
        raise ValueError(f"Expected maps with shape [N,H,W], got {maps.shape}")
    valid = maps > 0
    defect = maps == 2
    valid_count = valid.sum(axis=(1, 2)).astype(np.float32)
    defect_count = defect.sum(axis=(1, 2)).astype(np.float32)
    defect_float = defect.astype(np.float32)
    x_grid, y_grid, region_masks = _geometry_masks(maps.shape[1], maps.shape[2])

    features = np.empty((len(maps), len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = valid_count / float(maps.shape[1] * maps.shape[2])
    features[:, 1] = _safe_divide(defect_count, valid_count)
    center_x = _safe_divide((defect_float * x_grid).sum(axis=(1, 2)), defect_count)
    center_y = _safe_divide((defect_float * y_grid).sum(axis=(1, 2)), defect_count)
    features[:, 2] = center_x
    features[:, 3] = center_y
    centered_x = x_grid[None, :, :] - center_x[:, None, None]
    centered_y = y_grid[None, :, :] - center_y[:, None, None]
    features[:, 4] = np.sqrt(
        _safe_divide((defect_float * centered_x**2).sum(axis=(1, 2)), defect_count)
    )
    features[:, 5] = np.sqrt(
        _safe_divide((defect_float * centered_y**2).sum(axis=(1, 2)), defect_count)
    )
    features[:, 6] = _safe_divide(
        (defect_float * centered_x * centered_y).sum(axis=(1, 2)), defect_count
    )

    feature_index = len(BASE_FEATURE_NAMES)
    for region_mask in region_masks:
        region_valid = (valid & region_mask).sum(axis=(1, 2)).astype(np.float32)
        region_defect = (defect & region_mask).sum(axis=(1, 2)).astype(np.float32)
        features[:, feature_index] = _safe_divide(region_defect, region_valid)
        feature_index += 1
    return features


def _chunks(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def create_feature_cache(
    processed_dir: Path,
    cache_path: Path,
    batch_size: int = 2048,
) -> Path:
    processed_dir = Path(processed_dir)
    cache_path = Path(cache_path)
    images_path = processed_dir / "images.npy"
    if not images_path.exists():
        raise FileNotFoundError(f"Processed images are missing: {images_path}")
    images = np.load(images_path, mmap_mode="r")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(images), len(FEATURE_NAMES)),
    )
    all_indices = np.arange(len(images), dtype=np.int64)
    processed = 0
    for batch_indices in _chunks(all_indices, batch_size):
        cache[batch_indices] = extract_spatial_features(images[batch_indices])
        processed += len(batch_indices)
        if processed % 20_000 < batch_size:
            print(f"Extracted features for {processed:,}/{len(images):,} wafer maps")
    cache.flush()
    del cache
    with cache_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_count": len(FEATURE_NAMES),
                "feature_names": FEATURE_NAMES,
                "sample_count": len(images),
            },
            handle,
            indent=2,
        )
    return cache_path


def load_or_create_feature_cache(
    processed_dir: Path,
    cache_path: Path,
    batch_size: int = 2048,
    force: bool = False,
) -> np.ndarray:
    cache_path = Path(cache_path)
    labels_path = Path(processed_dir) / "labels.npy"
    expected_samples = len(np.load(labels_path, mmap_mode="r"))
    if cache_path.exists() and not force:
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == (expected_samples, len(FEATURE_NAMES)):
            return cached
        raise ValueError(
            f"Feature cache has shape {cached.shape}; expected "
            f"{(expected_samples, len(FEATURE_NAMES))}. Use force=true to rebuild it."
        )
    create_feature_cache(processed_dir, cache_path, batch_size=batch_size)
    return np.load(cache_path, mmap_mode="r")
