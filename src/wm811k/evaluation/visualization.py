"""Evaluation figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:,}",
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
