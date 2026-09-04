from __future__ import annotations

from pathlib import Path
from typing import Any, overload

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {source}")
    data["_config_path"] = str(source.resolve())
    return data


@overload
def resolve_from_config(config: dict[str, Any], value: None) -> None: ...


@overload
def resolve_from_config(config: dict[str, Any], value: str | Path) -> Path: ...


def resolve_from_config(
    config: dict[str, Any], value: str | Path | None
) -> Path | None:
    """Resolve a configured path against the directory containing its YAML file.

    ``None`` is preserved for optional path fields.  Every non-``None`` result is
    absolute, including paths that were already absolute in the YAML.
    """
    if value is None:
        return None
    base = Path(config["_config_path"]).parent
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()
