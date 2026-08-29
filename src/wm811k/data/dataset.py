"""Memory-mapped dataset and DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wm811k.data.transforms import WaferMapTransform


class WaferMapDataset(Dataset):
    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        indices: Optional[np.ndarray] = None,
        augment: bool = False,
    ) -> None:
        self.images = np.load(Path(images_path), mmap_mode="r")
        self.labels = np.load(Path(labels_path), mmap_mode="r")
        if len(self.images) != len(self.labels):
            raise ValueError("images.npy and labels.npy have different lengths")
        self.indices = (
            np.arange(len(self.labels), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.transform = WaferMapTransform(augment=augment)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        image = self.transform(self.images[index])
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, label


def build_dataloaders(
    processed_dir: Path,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    processed_dir = Path(processed_dir)
    images_path = processed_dir / "images.npy"
    labels_path = processed_dir / "labels.npy"
    splits_path = processed_dir / "splits.npz"
    for path in (images_path, labels_path, splits_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Processed file is missing: {path}. Run `python main.py preprocess` first."
            )

    splits = np.load(splits_path)
    loaders: Dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        dataset = WaferMapDataset(
            images_path,
            labels_path,
            indices=splits[split_name],
            augment=split_name == "train",
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=split_name == "train",
        )
    return loaders
