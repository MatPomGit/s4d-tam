from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10 compatibility
    import tomli as tomllib

SHA256 = re.compile(r"^[a-f0-9]{64}$")
DIGEST = re.compile(r"@sha256:[a-f0-9]{64}$")
EXACT_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+==[^ ;]+(?:\s+--hash=sha256:[a-f0-9]{64})+$")


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_release(root: str | Path) -> list[str]:
    """Return all static release-integrity violations (an empty list means releasable)."""
    root = Path(root).resolve()
    errors: list[str] = []
    release = _yaml(root / "release/version.yaml")
    version = str(release["release_version"])
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    identities = {
        "pyproject.toml": str(pyproject["project"]["version"]),
        "CITATION.cff": str(_yaml(root / "CITATION.cff")["version"]),
        "containers/images.yaml": str(_yaml(root / "containers/images.yaml")["release_version"]),
    }
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    identities["CHANGELOG.md"] = version if re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog, re.M) else "missing"
    for source, found in identities.items():
        if found != version:
            errors.append(f"version mismatch in {source}: {found!r} != {version!r}")

    lock = root / "containers/requirements.lock"
    for number, raw in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if line and not line.startswith("#") and not EXACT_REQUIREMENT.fullmatch(line):
            errors.append(f"unpinned or unhashed dependency {lock.relative_to(root)}:{number}")
    for dockerfile in sorted((root / "containers").glob("*.Dockerfile")):
        for number, raw in enumerate(dockerfile.read_text(encoding="utf-8").splitlines(), 1):
            if raw.startswith("FROM ") and not DIGEST.search(raw.split()[1]):
                errors.append(f"moving container base at {dockerfile.relative_to(root)}:{number}")
        if f'org.opencontainers.image.version="{version}"' not in dockerfile.read_text(encoding="utf-8"):
            errors.append(f"container version mismatch: {dockerfile.relative_to(root)}")

    image_inventory = _yaml(root / "containers/images.yaml")
    for name, image in image_inventory.get("images", {}).items():
        if str(image.get("tag")) in {"latest", "main", "master", "nightly"}:
            errors.append(f"moving release-image tag: {name}")
        dockerfile = root / str(image.get("dockerfile", ""))
        base = str(image.get("base_image", ""))
        if not DIGEST.search(base):
            errors.append(f"moving base image in inventory: {name}")
        elif dockerfile.is_file() and f"FROM {base}" not in dockerfile.read_text(encoding="utf-8"):
            errors.append(f"base-image inventory mismatch: {name}")

    for path in sorted((root / "manifests/datasets").glob("*.yaml")):
        item = _yaml(path)
        if str(item.get("schema_version")) != str(release["manifest_schema_version"]):
            errors.append(f"manifest schema version mismatch: {path.relative_to(root)}")
        if str(item.get("release_version")) != version:
            errors.append(f"manifest release version mismatch: {path.relative_to(root)}")
        if not item.get("license", {}).get("spdx") or not item.get("license", {}).get("text_or_url"):
            errors.append(f"missing manifest license: {path.relative_to(root)}")
        members = [member for split in item.get("splits", {}).values() for member in split]
        if len(members) != len(set(members)):
            errors.append(f"sequence occurs in multiple splits: {path.relative_to(root)}")
        for record in [*item.get("files", []), *item.get("calibrations", [])]:
            file_path = root / str(record.get("path", record.get("file", "")))
            expected = str(record.get("sha256", ""))
            if not file_path.is_file() or not SHA256.fullmatch(expected) or _digest(file_path) != expected:
                errors.append(f"checksum mismatch: {file_path.relative_to(root)}")
            elif file_path.stat().st_size != record.get("bytes"):
                errors.append(f"byte-count mismatch: {file_path.relative_to(root)}")
        for tool in item.get("conversion_tools", []):
            if not DIGEST.search(str(tool.get("container", ""))):
                errors.append(f"moving conversion container in {path.relative_to(root)}")

    for path in sorted((root / "weights").glob("*/metadata.yaml")):
        item = _yaml(path)
        if str(item.get("format_version")) != str(release["weights_format_version"]):
            errors.append(f"weights format version mismatch: {path.relative_to(root)}")
        if str(item.get("release_version")) != version:
            errors.append(f"weights release version mismatch: {path.relative_to(root)}")
        if not item.get("license", {}).get("spdx") or not item.get("license", {}).get("text_or_url"):
            errors.append(f"missing weights license: {path.relative_to(root)}")
        if str(item.get("architecture", {}).get("version")) != str(release["architecture_version"]):
            errors.append(f"architecture version mismatch: {path.relative_to(root)}")
        records = [item["weights"], item["architecture"], item["training"]]
        for record in records:
            file_path = root / str(record.get("file", record.get("config", "")))
            expected = str(record.get("sha256", record.get("config_sha256", "")))
            if not file_path.is_file() or not SHA256.fullmatch(expected) or _digest(file_path) != expected:
                errors.append(f"checksum mismatch: {file_path.relative_to(root)}")
        weights = item.get("weights", {})
        weights_file = root / str(weights.get("file", ""))
        if weights_file.is_file() and weights_file.stat().st_size != weights.get("bytes"):
            errors.append(f"byte-count mismatch: {weights_file.relative_to(root)}")
        for source in item.get("data_provenance", []):
            manifest = root / str(source["manifest"])
            if not manifest.is_file() or _digest(manifest) != source.get("manifest_sha256"):
                errors.append(f"checksum mismatch: {manifest.relative_to(root)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate S4D-TAM release metadata")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    errors = validate_release(args.root)
    if errors:
        print("Release validation failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    print("Release metadata and artifacts are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
