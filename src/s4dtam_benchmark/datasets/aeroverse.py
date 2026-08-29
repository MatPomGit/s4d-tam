"""Release- and license-gated AeroVerse adapter."""
from __future__ import annotations

import json
from pathlib import Path

from s4dtam_benchmark.datasets.manifest import ManifestDataset


class AeroVerseDataset(ManifestDataset):
    def __init__(self, root: str | Path, *, required_version: str,
                 accepted_license: str, manifest: str | Path | None = None):
        super().__init__("aeroverse", root, manifest)
        self.required_version = required_version
        self.accepted_license = accepted_license

    def sequences(self):
        if not self.manifest.exists():
            raise FileNotFoundError(
                f"AeroVerse data unavailable: complete download and manifest required at {self.manifest}"
            )
        spec = json.loads(self.manifest.read_text(encoding="utf-8"))
        if spec.get("dataset_version") != self.required_version:
            raise ValueError(f"AeroVerse release mismatch: required {self.required_version!r}, "
                             f"found {spec.get('dataset_version')!r}")
        license_spec = spec.get("license", {})
        if license_spec.get("id") != self.accepted_license or license_spec.get("accepted") is not True:
            raise PermissionError("AeroVerse license has not been explicitly accepted for this release")
        missing = [item["file"] for item in spec.get("sequences", [])
                   if not (self.root / item["file"]).is_file()]
        if not spec.get("sequences") or missing:
            detail = ", ".join(missing) if missing else "no sequences listed"
            raise FileNotFoundError(f"AeroVerse data unavailable or incomplete: {detail}")
        yield from super().sequences()
