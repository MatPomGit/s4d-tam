#!/usr/bin/env python3
"""Build a deterministic source archive and DOI-deposit metadata without minting a DOI."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VERSION = str(yaml.safe_load((ROOT / "release/version.yaml").read_text())["release_version"])
DIST = ROOT / "dist"
ARCHIVE = DIST / f"s4dtam-benchmark-{VERSION}.tar.gz"


def main() -> None:
    subprocess.run(["python", "-m", "s4dtam_benchmark.release", str(ROOT)], check=True)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=True)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    DIST.mkdir(exist_ok=True)
    with ARCHIVE.open("wb") as raw:
        import gzip

        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for relative in sorted(tracked):
                    info = archive.gettarinfo(ROOT / relative, arcname=f"s4dtam-benchmark-{VERSION}/{relative}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    with (ROOT / relative).open("rb") as source:
                        archive.addfile(info, source)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    metadata = yaml.safe_load((ROOT / ".zenodo.json.in").read_text())
    metadata["version"] = VERSION
    metadata["files"] = [{"name": ARCHIVE.name, "sha256": digest}]
    (DIST / "repository-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    release_metadata = {
        "schema_version": 1,
        "release_version": VERSION,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_archive": ARCHIVE.name,
        "source_archive_sha256": digest,
        "doi": None,
        "doi_status": "unregistered",
        "container_images": {
            "benchmark": {"digest": None, "status": "record after reproducible build"},
            "baselines": {"digest": None, "status": "record after reproducible build"}
        },
    }
    (DIST / "release-metadata.json").write_text(json.dumps(release_metadata, indent=2) + "\n")
    (DIST / "SHA256SUMS").write_text(f"{digest}  {ARCHIVE.name}\n")


if __name__ == "__main__":
    main()
