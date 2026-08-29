"""Classification metric calculation."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    labels = list(range(len(class_names)))
    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)
    return {
        "accuracy": float(accuracy_score(targets_array, predictions_array)),
        "balanced_accuracy": float(balanced_accuracy_score(targets_array, predictions_array)),
        "macro_f1": float(
            f1_score(targets_array, predictions_array, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                targets_array,
                predictions_array,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            targets_array,
            predictions_array,
            labels=labels,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets_array, predictions_array, labels=labels
        ).tolist(),
    }
