from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm811k.data.labels import normalize_label, unwrap_scalar  # noqa: E402
from wm811k.data.preprocess import resize_wafer_map  # noqa: E402
from wm811k.data.split import create_grouped_splits  # noqa: E402
from wm811k.data.transforms import WaferMapTransform  # noqa: E402
from wm811k.baselines.features import FEATURE_NAMES, extract_spatial_features  # noqa: E402


class LabelTests(unittest.TestCase):
    def test_array_wrapped_label(self):
        value = np.array([["Edge-Ring"]])
        self.assertEqual(unwrap_scalar(value), "Edge-Ring")
        self.assertEqual(normalize_label(value), "Edge-Ring")

    def test_empty_label(self):
        self.assertIsNone(normalize_label(np.array([])))


class TransformTests(unittest.TestCase):
    def test_two_channel_encoding(self):
        wafer = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        tensor = WaferMapTransform(augment=False)(wafer)
        self.assertEqual(tuple(tensor.shape), (2, 2, 2))
        np.testing.assert_array_equal(tensor[0].numpy(), [[0, 1], [1, 1]])
        np.testing.assert_array_equal(tensor[1].numpy(), [[0, 0], [1, 0]])

    def test_resize_preserves_aspect_ratio_with_zero_padding(self):
        wafer = np.ones((2, 4), dtype=np.uint8)
        resized = resize_wafer_map(wafer, image_size=4, preserve_aspect_ratio=True)
        np.testing.assert_array_equal(resized[0], np.zeros(4, dtype=np.uint8))
        np.testing.assert_array_equal(resized[-1], np.zeros(4, dtype=np.uint8))
        np.testing.assert_array_equal(resized[1:3], np.ones((2, 4), dtype=np.uint8))


class FeatureTests(unittest.TestCase):
    def test_spatial_features_are_finite_and_distinguish_location(self):
        center = np.ones((8, 8), dtype=np.uint8)
        center[3:5, 3:5] = 2
        edge = np.ones((8, 8), dtype=np.uint8)
        edge[0:2, 0:2] = 2
        features = extract_spatial_features(np.stack((center, edge)))
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features[0, 1]), float(features[1, 1]))
        self.assertGreater(abs(float(features[1, 2])), abs(float(features[0, 2])))


class SplitTests(unittest.TestCase):
    def test_group_isolation(self):
        groups = np.repeat(np.array([f"lot-{index}" for index in range(30)]), 6)
        labels = np.tile(np.arange(6), 30)
        splits = create_grouped_splits(labels, groups, candidates=8)
        split_groups = {name: set(groups[indices]) for name, indices in splits.items()}
        self.assertTrue(split_groups["train"].isdisjoint(split_groups["val"]))
        self.assertTrue(split_groups["train"].isdisjoint(split_groups["test"]))
        self.assertTrue(split_groups["val"].isdisjoint(split_groups["test"]))
        combined = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(labels)))


if __name__ == "__main__":
    unittest.main()
