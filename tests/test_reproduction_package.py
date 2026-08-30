import hashlib
from pathlib import Path

import pytest

from s4dtam_benchmark.reproduction import verify_reproduction_package


def _package(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    payload = tmp_path / "input.txt"
    payload.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "checksums.sha256").write_text(f"{digest}  input.txt\n", encoding="utf-8")
    spec = {
        "required_paths": ["input.txt"],
        "required_metadata": ["checksums.sha256"],
        "forbidden_paths": [".git", "outputs"],
    }
    return tmp_path, spec


def test_verifies_complete_package(tmp_path: Path) -> None:
    root, spec = _package(tmp_path)
    verify_reproduction_package(root, spec)


def test_rejects_modified_input(tmp_path: Path) -> None:
    root, spec = _package(tmp_path)
    (root / "input.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_reproduction_package(root, spec)


def test_rejects_cache_or_output(tmp_path: Path) -> None:
    root, spec = _package(tmp_path)
    (root / "outputs").mkdir()
    with pytest.raises(ValueError, match="forbidden package path"):
        verify_reproduction_package(root, spec)


def test_rejects_required_file_missing_from_inventory(tmp_path: Path) -> None:
    root, spec = _package(tmp_path)
    extra = root / "metadata.json"
    extra.write_text("{}\n", encoding="utf-8")
    spec["required_metadata"] = ["checksums.sha256", "metadata.json"]
    with pytest.raises(ValueError, match="required file is not checksummed"):
        verify_reproduction_package(root, spec)
