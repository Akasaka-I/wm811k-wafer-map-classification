"""Wafer-map tensor transforms that preserve discrete die states."""

from __future__ import annotations

import numpy as np
import torch


class WaferMapTransform:
    """Convert a {0,1,2} wafer map to valid-die and failed-die channels."""

    def __init__(self, augment: bool = False) -> None:
        self.augment = augment

    def __call__(self, wafer_map: np.ndarray) -> torch.Tensor:
        discrete = torch.from_numpy(np.array(wafer_map, dtype=np.uint8, copy=True))
        valid_mask = discrete.gt(0)
        defect_mask = discrete.eq(2)
        tensor = torch.stack((valid_mask, defect_mask), dim=0).float()

        if self.augment:
            rotations = int(torch.randint(0, 4, ()).item())
            tensor = torch.rot90(tensor, rotations, dims=(-2, -1))
            if bool(torch.rand(()) < 0.5):
                tensor = torch.flip(tensor, dims=(-1,))
            if bool(torch.rand(()) < 0.5):
                tensor = torch.flip(tensor, dims=(-2,))
        return tensor.contiguous()
