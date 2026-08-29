"""Normalization for WM-811K's array-wrapped metadata values."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from wm811k.data import CLASS_NAMES


_CANONICAL = {name.lower(): name for name in CLASS_NAMES}


def unwrap_scalar(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return unwrap_scalar(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return unwrap_scalar(value[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_label(value: Any) -> Optional[str]:
    scalar = unwrap_scalar(value)
    if scalar is None:
        return None
    label = str(scalar).strip()
    if not label or label.lower() in {"nan", "[]"}:
        return None
    return _CANONICAL.get(label.lower())


def normalize_group(value: Any, fallback: str) -> str:
    scalar = unwrap_scalar(value)
    if scalar is None:
        return fallback
    group = str(scalar).strip()
    return group if group else fallback
