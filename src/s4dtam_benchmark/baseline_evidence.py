"""Validation and immutable evidence generation for reproduced external baselines."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s4dtam_benchmark.algorithms.external import WRAPPERS, parse_external_artifact
from s4dtam_benchmark.config import load_yaml

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class BaselineEvidenceSummary:
    baseline: str
    dataset: str
    sequences: int
    evidence_path: Path
    evidence_sha256: str


def validate_and_freeze_baseline_evidence(
    *,
    baseline: str,
    dataset: str,
    sequence_list: str | Path,
    result_root: str | Path,
    config_path: str | Path,
    run_metadata_path: str | Path,
    output_dir: str | Path,
) -> BaselineEvidenceSummary:
    """Validate one frozen baseline cohort and emit an immutable evidence manifest.

    The generated JSON is deterministic for identical inputs: validation time is deliberately not
    inserted into the hashed payload. Any execution timestamp that is scientifically relevant must
    be recorded in the run metadata produced by the real baseline execution environment.
    """
    if baseline not in WRAPPERS:
        raise ValueError(f"Unsupported external baseline: {baseline}")

    config_file = Path(config_path)
    config = load_yaml(config_file)
    if config.get("name") != baseline or config.get("type") != "external_artifact":
        raise ValueError("Baseline config name/type does not match requested external baseline")
    wrapper = WRAPPERS[baseline]
    if config.get("revision") != wrapper.commit:
        raise ValueError("Baseline config revision does not match the executable wrapper contract")
    if config.get("container") != wrapper.container:
        raise ValueError("Baseline config container does not match the executable wrapper contract")

    sequence_list_file = Path(sequence_list)
    sequence_ids = [
        line.strip() for line in sequence_list_file.read_text(encoding="utf-8").splitlines()
    ]
    sequence_ids = [sequence_id for sequence_id in sequence_ids if sequence_id]
    if not sequence_ids:
        raise ValueError("Baseline sequence list is empty")
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("Baseline sequence list contains duplicate sequence IDs")

    run_metadata_file = Path(run_metadata_path)
    metadata: dict[str, Any] = json.loads(run_metadata_file.read_text(encoding="utf-8"))
    _validate_run_metadata(metadata, baseline=baseline, dataset=dataset, config=config)

    dataset_result_root = Path(result_root) / dataset
    expected = {f"{sequence_id}.npz" for sequence_id in sequence_ids}
    present = (
        {path.name for path in dataset_result_root.glob("*.npz")}
        if dataset_result_root.is_dir()
        else set()
    )
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        raise FileNotFoundError(f"Missing baseline artifacts for frozen cohort: {', '.join(missing)}")
    if unexpected:
        raise ValueError(
            "Baseline result directory contains artifacts outside the frozen cohort: "
            + ", ".join(unexpected)
        )

    artifact_hashes: dict[str, str] = {}
    sample_counts: dict[str, int] = {}
    for sequence_id in sequence_ids:
        artifact_path = dataset_result_root / f"{sequence_id}.npz"
        result = parse_external_artifact(artifact_path, baseline)
        artifact_hashes[sequence_id] = _sha256_file(artifact_path)
        sample_counts[sequence_id] = int(result.timestamps.size)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema": "s4dtam-baseline-reproduction-evidence/v1",
        "baseline": baseline,
        "dataset": dataset,
        "wrapper": {
            "upstream": wrapper.upstream,
            "revision": wrapper.commit,
            "container": wrapper.container,
            "output_format": wrapper.output_format,
        },
        "input_manifest_sha256": metadata["input_manifest_sha256"],
        "sequence_list_sha256": _sha256_file(sequence_list_file),
        "baseline_config_sha256": _sha256_file(config_file),
        "run_metadata_sha256": _sha256_file(run_metadata_file),
        "run_metadata": metadata,
        "sequence_ids": sequence_ids,
        "artifact_sha256": artifact_hashes,
        "artifact_sample_counts": sample_counts,
        "validation": {
            "normalized_artifact_contract": "passed",
            "frozen_cohort_exact_match": "passed",
            "pinned_revision_match": "passed",
            "pinned_container_match": "passed",
        },
    }
    evidence_path = output / f"{baseline}-{dataset}-evidence.json"
    evidence_text = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    evidence_path.write_text(evidence_text, encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
    (output / f"{baseline}-{dataset}-evidence.sha256").write_text(
        f"{evidence_hash}  {evidence_path.name}\n", encoding="utf-8"
    )
    return BaselineEvidenceSummary(
        baseline=baseline,
        dataset=dataset,
        sequences=len(sequence_ids),
        evidence_path=evidence_path,
        evidence_sha256=evidence_hash,
    )


def _validate_run_metadata(
    metadata: dict[str, Any], *, baseline: str, dataset: str, config: dict[str, Any]
) -> None:
    required = {
        "baseline",
        "dataset",
        "revision",
        "container",
        "input_manifest_sha256",
        "hardware",
        "command",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Baseline run metadata missing fields: {', '.join(missing)}")
    if metadata["baseline"] != baseline or metadata["dataset"] != dataset:
        raise ValueError("Baseline run metadata baseline/dataset does not match requested cohort")
    if metadata["revision"] != config.get("revision"):
        raise ValueError("Baseline run metadata revision does not match pinned config")
    if metadata["container"] != config.get("container"):
        raise ValueError("Baseline run metadata container does not match pinned config")
    if not _SHA256_RE.fullmatch(str(metadata["input_manifest_sha256"])):
        raise ValueError("Baseline run metadata input_manifest_sha256 must be a lowercase SHA-256")
    if not isinstance(metadata["hardware"], dict) or not metadata["hardware"]:
        raise ValueError("Baseline run metadata hardware must be a non-empty object")
    if not isinstance(metadata["command"], str) or not metadata["command"].strip():
        raise ValueError("Baseline run metadata command must record the executed command")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
