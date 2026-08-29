"""Minimal Grad-CAM implementation for convolutional classifiers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as functional
from torch import nn


class GradCAM:
    """Generate a normalized class activation map for one input image."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._capture_activations)

    def _capture_activations(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradients: torch.Tensor) -> None:
        self.gradients = gradients

    def generate(self, inputs: torch.Tensor, target_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or len(inputs) != 1:
            raise ValueError("GradCAM.generate expects an input batch containing exactly one image")
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        logits[0, target_class].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        activation_map = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        activation_map = functional.interpolate(
            activation_map,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        minimum = activation_map.min()
        maximum = activation_map.max()
        activation_map = (activation_map - minimum) / (maximum - minimum).clamp_min(1e-8)
        return activation_map.detach(), logits.detach()

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.close()
