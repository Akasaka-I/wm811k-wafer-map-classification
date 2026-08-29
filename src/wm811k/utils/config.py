"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def require_keys(mapping: Dict[str, Any], *keys: str, context: str = "config") -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing {context} keys: {', '.join(missing)}")
