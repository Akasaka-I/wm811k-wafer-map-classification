"""Raw WM-811K dataset audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from wm811k.data.labels import normalize_label


@dataclass
class DatasetSummary:
    path: str
    rows: int
    columns: List[str]
    memory_gib: float
    labeled_rows: int
    unlabeled_rows: int
    class_counts: Dict[str, int]
    wafer_shape_counts: Dict[str, int]
    unique_lots: int

    def to_text(self) -> str:
        classes = ", ".join(f"{key}={value}" for key, value in self.class_counts.items())
        return (
            f"Rows: {self.rows:,}\n"
            f"Columns: {', '.join(self.columns)}\n"
            f"DataFrame memory: {self.memory_gib:.2f} GiB\n"
            f"Labeled / unlabeled: {self.labeled_rows:,} / {self.unlabeled_rows:,}\n"
            f"Unique lots: {self.unique_lots:,}\n"
            f"Classes: {classes}"
        )


def _shape_key(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return f"{value.shape[0]}x{value.shape[1]}"
    return "invalid"


def inspect_raw_dataset(data_path: Path, output_path: Path) -> DatasetSummary:
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset does not exist: {data_path}")

    frame = pd.read_pickle(data_path)
    required = {"waferMap", "failureType"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Raw dataset is missing columns: {', '.join(missing)}")

    labels = frame["failureType"].map(normalize_label)
    class_counts = labels.dropna().value_counts().sort_index().astype(int).to_dict()
    shape_counts = frame["waferMap"].map(_shape_key).value_counts().head(25).astype(int).to_dict()
    unique_lots = int(frame["lotName"].nunique()) if "lotName" in frame.columns else 0
    memory_gib = float(frame.memory_usage(index=True, deep=True).sum() / (1024**3))

    summary = DatasetSummary(
        path=str(data_path.resolve()),
        rows=int(len(frame)),
        columns=[str(column) for column in frame.columns],
        memory_gib=round(memory_gib, 4),
        labeled_rows=int(labels.notna().sum()),
        unlabeled_rows=int(labels.isna().sum()),
        class_counts=class_counts,
        wafer_shape_counts=shape_counts,
        unique_lots=unique_lots,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, ensure_ascii=False)
    return summary
