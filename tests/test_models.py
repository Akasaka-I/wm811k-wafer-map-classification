from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm811k.models.factory import available_models, build_model  # noqa: E402


class ModelFactoryTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(
            available_models(),
            ["convnext_tiny", "custom_cnn", "resnet18", "swin_tiny"],
        )

    def test_all_models_output_nine_logits(self):
        inputs = torch.randn(1, 2, 64, 64)
        for name in available_models():
            with self.subTest(model=name):
                model = build_model(
                    {
                        "name": name,
                        "input_channels": 2,
                        "num_classes": 9,
                        "pretrained": False,
                    }
                )
                model.eval()
                with torch.inference_mode():
                    outputs = model(inputs)
                self.assertEqual(tuple(outputs.shape), (1, 9))


if __name__ == "__main__":
    unittest.main()
