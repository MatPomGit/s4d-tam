from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes package root: {relative}")
    return path


def verify_reproduction_package(root: str | Path, spec: dict[str, Any]) -> None:
    """Verify the structure and SHA-256 inventory of a released input package."""
    package = Path(root).resolve()
    errors: list[str] = []
    required = [*spec.get("required_paths", []), *spec.get("required_metadata", [])]
    for relative in required:
        try:
            path = _inside(package, str(relative))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
    for relative in spec.get("forbidden_paths", []):
        if _inside(package, str(relative)).exists():
            errors.append(f"forbidden package path exists: {relative}")

    checksum_file = package / "checksums.sha256"
    inventoried: set[str] = set()
    if checksum_file.is_file():
        for line_number, raw in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            parts = raw.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"invalid checksum line {line_number}")
                continue
            expected, relative = parts[0], parts[1].lstrip("* ")
            inventoried.add(relative)
            try:
                path = _inside(package, relative)
            except ValueError as error:
                errors.append(str(error))
                continue
            if not path.is_file():
                errors.append(f"checksummed file is missing: {relative}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected.lower():
                errors.append(f"checksum mismatch: {relative}")
        for relative in required:
            if str(relative) != "checksums.sha256" and str(relative) not in inventoried:
                errors.append(f"required file is not checksummed: {relative}")
    if errors:
        raise ValueError("Invalid reproduction package:\n- " + "\n- ".join(errors))
