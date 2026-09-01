"""Import experiment results into PostgreSQL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from wm811k.storage.database import connect_database
from wm811k.data import CLASS_NAMES

MODEL_REQUIRED_FIELDS = {
    "name",
    "kind",
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "validation_accuracy",
    "parameters",
    "history_path",
}

ERROR_REQUIRED_FIELDS = {
    "dataset_position",
    "global_index",
    "lot",
    "true_class",
    "predicted_class",
    "confidence",
}

UPSERT_MODEL_SQL = """
INSERT INTO model_experiments (
    model_name,
    model_kind,
    validation_macro_f1,
    validation_balanced_accuracy,
    validation_accuracy,
    parameter_count,
    history_path
)
VALUES(
    %(model_name)s,
    %(model_kind)s,
    %(validation_macro_f1)s,
    %(validation_balanced_accuracy)s,
    %(validation_accuracy)s,
    %(parameter_count)s,
    %(history_path)s
)
ON CONFLICT (model_name)
DO UPDATE SET
    model_kind = EXCLUDED.model_kind,
    validation_macro_f1 = EXCLUDED.validation_macro_f1,
    validation_balanced_accuracy = EXCLUDED.validation_balanced_accuracy,
    validation_accuracy = EXCLUDED.validation_accuracy,
    parameter_count = EXCLUDED.parameter_count,
    history_path = EXCLUDED.history_path
"""

UPSERT_CLASS_METRIC_SQL = """
INSERT INTO class_metrics (
    experiment_id,
    split_name,
    class_name,
    precision_score,
    recall_score,
    f1_score,
    support_count
)
VALUES (
    %(experiment_id)s,
    %(split_name)s,
    %(class_name)s,
    %(precision_score)s,
    %(recall_score)s,
    %(f1_score)s,
    %(support_count)s
)
ON CONFLICT (experiment_id, split_name, class_name)
DO UPDATE SET
    precision_score = EXCLUDED.precision_score,
    recall_score = EXCLUDED.recall_score,
    f1_score = EXCLUDED.f1_score,
    support_count = EXCLUDED.support_count
"""

UPSERT_PREDICTION_ERROR_SQL = """
INSERT INTO prediction_errors (
    experiment_id,
    split_name,
    dataset_position,
    global_index,
    lot_name,
    true_class,
    predicted_class,
    confidence
)
VALUES (
    %(experiment_id)s,
    %(split_name)s,
    %(dataset_position)s,
    %(global_index)s,
    %(lot_name)s,
    %(true_class)s,
    %(predicted_class)s,
    %(confidence)s
)
ON CONFLICT (experiment_id, split_name, global_index)
DO UPDATE SET
    dataset_position = EXCLUDED.dataset_position,
    lot_name = EXCLUDED.lot_name,
    true_class = EXCLUDED.true_class,
    predicted_class = EXCLUDED.predicted_class,
    confidence = EXCLUDED.confidence
"""

def load_model_experiments(path: Path) -> list[dict[str, Any]]:
    """Load and Validate model-comparison records from JSON."""

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError("Model-comparison JSON must contain a list")

    transformed_records = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object")

        missing_fields = MODEL_REQUIRED_FIELDS.difference(record)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Record {index} is missing fields: {missing}")

        transformed_records.append(
            {
                "model_name": str(record["name"]),
                "model_kind": str(record["kind"]),
                "validation_macro_f1": float(record["validation_macro_f1"]),
                "validation_balanced_accuracy": float(record["validation_balanced_accuracy"]),
                "validation_accuracy": float(record["validation_accuracy"]),
                "parameter_count": (
                    int(record["parameters"])
                    if record["parameters"] is not None
                    else None
                ),
                "history_path": (
                    str(record["history_path"])
                    if record["history_path"] is not None
                    else None
                ),
            }
        )
    return transformed_records

def load_class_metrics(
    path: Path,
    model_name: str,
    split_name: str,
) -> list[dict[str, Any]]:
    """Load per-class metrics from an evaluation JSON file."""

    if split_name not in {"validation", "test"}:
        raise ValueError(
            f"Unsupported split name:{split_name}. "
            "Expected 'validation' or 'test'."
        )
    with path.open("r", encoding='utf-8') as handle:
        metrics = json.load(handle)

    if not isinstance(metrics, dict):
        raise ValueError("Evaluation JSON must contain an object")

    report = metrics.get("classification_report")
    if not isinstance(report, dict):
        raise ValueError("Evaluation JSON is missing classification_report")

    transformed_records = []

    for class_name in CLASS_NAMES:
        class_result = report.get(class_name)

        if not isinstance(class_result, dict):
            raise ValueError(
                f"classification_report is missing class: {class_name}"
            )

        required_fields = {"precision", "recall", "f1-score", "support"}
        missing_fields = required_fields.difference(class_result)

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Class {class_name} is missing fields: {missing}"
            )

        transformed_records.append(
            {
                "model_name": model_name,
                "split_name": split_name,
                "class_name": class_name,
                "precision_score": float(class_result["precision"]),
                "recall_score": float(class_result["recall"]),
                "f1_score": float(class_result["f1-score"]),
                "support_count": int(class_result["support"]),
            }
        )
    return transformed_records

def load_prediction_errors(
    path: Path,
    model_name: str,
    split_name: str,
) -> list[dict[str, Any]]:
    """Load prediction errors from JSON."""

    if split_name not in {"validation", "test"}:
        raise ValueError(
            f"Unsupported split name: {split_name}. "
            "Expected 'validation' or 'test'."
        )

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError("Prediction-error JSON must contain a list")

    transformed_records = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Error record {index} must be a JSON object")

        missing_fields = ERROR_REQUIRED_FIELDS.difference(record)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Error record {index} is missing fields: {missing}"
            )

        true_class = str(record["true_class"])
        predicted_class = str(record["predicted_class"])
        confidence = float(record["confidence"])

        if true_class == predicted_class:
            raise ValueError(
                f"Error record {index} is not a misclassification"
            )

        if not 0 <= confidence <= 1:
            raise ValueError(
                f"Error record {index} has invalid confidence: {confidence}"
            )

        transformed_records.append(
            {
                "model_name": model_name,
                "split_name": split_name,
                "dataset_position": int(record["dataset_position"]),
                "global_index": int(record["global_index"]),
                "lot_name": str(record["lot"]),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
            }
        )

    return transformed_records

def import_model_experiments(records: list[dict[str, Any]]) -> int:
    """Insert or update model experiment records."""

    if not records:
        return 0

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(UPSERT_MODEL_SQL, records)

    return len(records)


def import_class_metrics(records: list[dict[str, Any]]) -> int:
    """Insert or update per-class evaluation metrics."""

    if not records:
        return 0

    model_names = {str(record["model_name"]) for record in records}

    if len(model_names) != 1:
        raise ValueError(
            "Class metric records must belong to exactly one model"
        )

    model_name = next(iter(model_names))

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM model_experiments
                WHERE model_name = %s
                """,
                (model_name,),
            )
            experiment = cursor.fetchone()

            if experiment is None:
                raise ValueError(
                    f"Model experiment does not exist: {model_name}"
                )

            experiment_id = int(experiment[0])

            database_records = [
                {
                    **record,
                    "experiment_id": experiment_id,
                }
                for record in records
            ]

            cursor.executemany(
                UPSERT_CLASS_METRIC_SQL,
                database_records,
            )
    return len(records)

def import_prediction_errors(records: list[dict[str, Any]]) -> int:
    """Insert or update prediction-error records."""

    if not records:
        return 0

    model_names = {str(record["model_name"]) for record in records}

    if len(model_names) != 1:
        raise ValueError(
            "Prediction-error records must belong to exactly one model"
        )

    model_name = next(iter(model_names))

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM model_experiments
                WHERE model_name = %s
                """,
                (model_name,),
            )
            experiment = cursor.fetchone()

            if experiment is None:
                raise ValueError(
                    f"Model experiment does not exist: {model_name}"
                )

            experiment_id = int(experiment[0])

            database_records = [
                {
                    **record,
                    "experiment_id": experiment_id,
                }
                for record in records
            ]

            cursor.executemany(
                UPSERT_PREDICTION_ERROR_SQL,
                database_records,
            )

    return len(records)