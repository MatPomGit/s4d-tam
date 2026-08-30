from pathlib import Path

from s4dtam_benchmark.release import validate_release


ROOT = Path(__file__).resolve().parents[1]


def test_repository_release_is_consistent() -> None:
    assert validate_release(ROOT) == []


def test_release_check_reports_moving_base(tmp_path: Path) -> None:
    dockerfile = ROOT / "containers/benchmark.Dockerfile"
    original = dockerfile.read_text(encoding="utf-8")
    try:
        dockerfile.write_text(original.replace("@sha256:" + "27f90d79cc85e9b7b2560063ef44fa0e9eaae7a7c3f5a9f74563065c5477cc24", ""))
        assert any("moving container base" in error for error in validate_release(ROOT))
    finally:
        dockerfile.write_text(original, encoding="utf-8")


def test_release_check_reports_changed_artifact() -> None:
    artifact = ROOT / "weights/fixture/model.bin"
    original = artifact.read_bytes()
    try:
        artifact.write_bytes(b"modified")
        assert any("checksum mismatch" in error for error in validate_release(ROOT))
    finally:
        artifact.write_bytes(original)


def test_release_check_reports_unpinned_dependency() -> None:
    lock = ROOT / "containers/requirements.lock"
    original = lock.read_text(encoding="utf-8")
    try:
        lock.write_text(original + "example>=1\n", encoding="utf-8")
        assert any("unpinned or unhashed dependency" in error for error in validate_release(ROOT))
    finally:
        lock.write_text(original, encoding="utf-8")


def test_release_check_reports_missing_license() -> None:
    manifest = ROOT / "manifests/datasets/synthetic-v1.yaml"
    original = manifest.read_text(encoding="utf-8")
    try:
        manifest.write_text(original.replace("  spdx: Apache-2.0", "  spdx: ''", 1), encoding="utf-8")
        assert any("missing manifest license" in error for error in validate_release(ROOT))
    finally:
        manifest.write_text(original, encoding="utf-8")
