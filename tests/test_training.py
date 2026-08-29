from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm811k.training.losses import balanced_class_weights  # noqa: E402
from wm811k.training.scheduler import build_scheduler  # noqa: E402


class ClassWeightTests(unittest.TestCase):
    def test_sqrt_weights_reduce_and_clip_extremes(self):
        labels = np.array([0] * 1000 + [1] * 10 + [2], dtype=np.int64)
        standard = balanced_class_weights(labels, 3)
        stable = balanced_class_weights(labels, 3, power=0.5, max_weight=10.0)
        self.assertGreater(float(standard.max()), 100.0)
        self.assertLessEqual(float(stable.max()), 10.0)
        self.assertGreater(float(stable[2]), float(stable[1]))


class SchedulerTests(unittest.TestCase):
    def test_warmup_cosine_starts_below_base_learning_rate(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = build_scheduler(
            optimizer,
            {"name": "warmup_cosine", "warmup_epochs": 2, "start_factor": 0.1},
            epochs=10,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        optimizer.step()
        scheduler.step()
        self.assertGreater(optimizer.param_groups[0]["lr"], 1e-5)


if __name__ == "__main__":
    unittest.main()
