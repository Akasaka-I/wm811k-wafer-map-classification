import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm811k.data import CLASS_NAMES  # noqa: E402
from wm811k.storage.importer import (  # noqa: E402
    load_class_metrics,
    load_model_experiments,
    load_prediction_errors,
)


class StorageLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_json(self, name, value):
        path = self.directory / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)
        return path

    def test_load_model_experiments(self):
        path = self.write_json(
            "models.json",
            [
                {
                    "name": "Test Model",
                    "kind": "deep",
                    "validation_macro_f1": 0.8,
                    "validation_balanced_accuracy": 0.85,
                    "validation_accuracy": 0.9,
                    "parameters": 12345,
                    "history_path": "history.json",
                }
            ],
        )

        records = load_model_experiments(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["model_name"], "Test Model")
        self.assertEqual(records[0]["parameter_count"], 12345)

    def test_load_class_metrics(self):
        report = {
            class_name: {
                "precision": 0.8,
                "recall": 0.7,
                "f1-score": 0.75,
                "support": 10,
            }
            for class_name in CLASS_NAMES
        }
        path = self.write_json(
            "metrics.json",
            {"classification_report": report},
        )

        records = load_class_metrics(path, "Test Model", "test")

        self.assertEqual(len(records), len(CLASS_NAMES))
        self.assertEqual(
            sum(record["support_count"] for record in records),
            10 * len(CLASS_NAMES),
        )

    def test_load_prediction_errors(self):
        path = self.write_json(
            "errors.json",
            [
                {
                    "dataset_position": 5,
                    "global_index": 100,
                    "lot": "lot1",
                    "true_class": "Center",
                    "predicted_class": "none",
                    "confidence": 0.95,
                }
            ],
        )

        records = load_prediction_errors(
            path,
            "Test Model",
            "test",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["lot_name"], "lot1")
        self.assertEqual(records[0]["confidence"], 0.95)

    def test_rejects_non_error_prediction(self):
        path = self.write_json(
            "not_an_error.json",
            [
                {
                    "dataset_position": 5,
                    "global_index": 100,
                    "lot": "lot1",
                    "true_class": "Center",
                    "predicted_class": "Center",
                    "confidence": 0.95,
                }
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "is not a misclassification",
        ):
            load_prediction_errors(path, "Test Model", "test")


if __name__ == "__main__":
    unittest.main()