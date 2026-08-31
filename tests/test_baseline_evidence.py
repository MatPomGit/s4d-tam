from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from s4dtam_benchmark.baseline_evidence import validate_and_freeze_baseline_evidence
from s4dtam_benchmark.config import load_yaml


def _artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        timestamps=np.array([0.0, 0.1]),
        estimated_positions=np.zeros((2, 3)),
        estimated_quaternions=np.array([[0.0, 0.0, 0.0, 1.0]] * 2),
        latency_ms=np.array([2.0, 2.5]),
        resource_peak_rss_mb=np.array(100.0),
        resource_cpu_time_s=np.array(1.0),
    )


def _metadata(path: Path, baseline: str, dataset: str, config_path: Path) -> None:
    config = load_yaml(config_path)
    path.write_text(
        json.dumps(
            {
                "baseline": baseline,
                "dataset": dataset,
                "revision": config["revision"],
                "container": config["container"],
                "input_manifest_sha256": "a" * 64,
                "hardware": {"cpu": "test-cpu", "ram_gb": 32},
                "command": "docker run pinned-image ...",
            }
        ),
        encoding="utf-8",
    )


def test_validate_and_freeze_baseline_evidence(tmp_path: Path) -> None:
    sequence_list = tmp_path / "sequence-list.txt"
    sequence_list.write_text("seq-a\nseq-b\n", encoding="utf-8")
    result_root = tmp_path / "results"
    _artifact(result_root / "tartanair" / "seq-a.npz")
    _artifact(result_root / "tartanair" / "seq-b.npz")
    config_path = Path("configs/algorithms/orb_slam3.yaml")
    metadata = tmp_path / "run.json"
    _metadata(metadata, "orb_slam3", "tartanair", config_path)

    summary = validate_and_freeze_baseline_evidence(
        baseline="orb_slam3",
        dataset="tartanair",
        sequence_list=sequence_list,
        result_root=result_root,
        config_path=config_path,
        run_metadata_path=metadata,
        output_dir=tmp_path / "evidence",
    )

    assert summary.sequences == 2
    assert len(summary.evidence_sha256) == 64
    evidence = json.loads(summary.evidence_path.read_text(encoding="utf-8"))
    assert evidence["validation"]["normalized_artifact_contract"] == "passed"
    assert set(evidence["artifact_sha256"]) == {"seq-a", "seq-b"}
    assert evidence["wrapper"]["revision"] == load_yaml(config_path)["revision"]


def test_baseline_evidence_rejects_artifact_outside_frozen_cohort(tmp_path: Path) -> None:
    sequence_list = tmp_path / "sequence-list.txt"
    sequence_list.write_text("seq-a\n", encoding="utf-8")
    result_root = tmp_path / "results"
    _artifact(result_root / "tartanair" / "seq-a.npz")
    _artifact(result_root / "tartanair" / "unexpected.npz")
    config_path = Path("configs/algorithms/orb_slam3.yaml")
    metadata = tmp_path / "run.json"
    _metadata(metadata, "orb_slam3", "tartanair", config_path)

    with pytest.raises(ValueError, match="outside the frozen cohort"):
        validate_and_freeze_baseline_evidence(
            baseline="orb_slam3",
            dataset="tartanair",
            sequence_list=sequence_list,
            result_root=result_root,
            config_path=config_path,
            run_metadata_path=metadata,
            output_dir=tmp_path / "evidence",
        )


def test_baseline_evidence_rejects_unpinned_run_metadata(tmp_path: Path) -> None:
    sequence_list = tmp_path / "sequence-list.txt"
    sequence_list.write_text("seq-a\n", encoding="utf-8")
    result_root = tmp_path / "results"
    _artifact(result_root / "tartanair" / "seq-a.npz")
    config_path = Path("configs/algorithms/orb_slam3.yaml")
    metadata = tmp_path / "run.json"
    _metadata(metadata, "orb_slam3", "tartanair", config_path)
    content = json.loads(metadata.read_text(encoding="utf-8"))
    content["container"] = "orbslam3:latest"
    metadata.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError, match="container does not match pinned config"):
        validate_and_freeze_baseline_evidence(
            baseline="orb_slam3",
            dataset="tartanair",
            sequence_list=sequence_list,
            result_root=result_root,
            config_path=config_path,
            run_metadata_path=metadata,
            output_dir=tmp_path / "evidence",
        )
