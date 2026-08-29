from pathlib import Path
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm811k.analysis.gradcam import GradCAM  # noqa: E402


class GradCAMTests(unittest.TestCase):
    def test_activation_map_is_finite_and_normalized(self):
        model = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 3),
        )
        inputs = torch.randn(1, 2, 16, 16)
        with GradCAM(model, model[0]) as gradcam:
            activation_map, logits = gradcam.generate(inputs, target_class=1)
        self.assertEqual(tuple(activation_map.shape), (16, 16))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertTrue(bool(torch.isfinite(activation_map).all()))
        self.assertGreaterEqual(float(activation_map.min()), 0.0)
        self.assertLessEqual(float(activation_map.max()), 1.0)
