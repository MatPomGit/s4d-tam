from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {source}")
    data["_config_path"] = str(source.resolve())
    return data


def resolve_from_config(config: dict[str, Any], value: str) -> Path:
    base = Path(config["_config_path"]).parent
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
