from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_declared_type_stubs_are_not_silenced_by_mypy() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    development_dependencies = config["project"]["optional-dependencies"]["dev"]
    overrides = config["tool"]["mypy"].get("overrides", [])

    assert any(dependency.startswith("pandas-stubs") for dependency in development_dependencies)
    assert any(dependency.startswith("types-PyYAML") for dependency in development_dependencies)

    ignored_modules = {
        module
        for override in overrides
        if override.get("ignore_missing_imports")
        for module in override["module"]
    }
    assert "pandas" not in ignored_modules
    assert "yaml" not in ignored_modules
