from __future__ import annotations

import re
from typing import Any

DATASETS = ("tartanair", "blackbird", "marsim", "aeroverse")
BASELINES = ("orb_slam3", "vins_mono", "fast_lio2", "lio_sam")
MODELS = (
    "full",
    "H1_no_semantics",
    "H2_no_temporal_state",
    "H3_no_calibrated_uncertainty",
    "H4_no_topology",
    "H5_no_reference_map",
    "H6_no_risk_prediction",
    "H7_no_token_lifecycle",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha(value: object, *, length: int, field: str) -> str:
    text = str(value or "").lower()
    pattern = _HEX40 if length == 40 else _HEX64
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase {length}-hex SHA")
    return text


def validate_confirmatory_freeze(config: dict[str, Any]) -> None:
    """Reject a confirmatory-study freeze unless every evidence surface is immutable."""
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("confirmatory freeze schema_version must be 1")
    if config.get("study_state") != "frozen":
        raise ValueError("study_state must be 'frozen' before confirmatory execution")

    _sha(config.get("code_commit"), length=40, field="code_commit")
    prereg = config.get("preregistration", {})
    _sha(prereg.get("sha256"), length=64, field="preregistration.sha256")
    if not str(prereg.get("external_timestamp", "")).strip():
        raise ValueError("preregistration.external_timestamp is required")

    datasets = config.get("datasets", {})
    for name in DATASETS:
        spec = datasets.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"missing frozen dataset entry: {name}")
        if not str(spec.get("release", "")).strip():
            raise ValueError(f"dataset {name} must declare release")
        _sha(spec.get("manifest_sha256"), length=64, field=f"datasets.{name}.manifest_sha256")
        _sha(spec.get("sequence_list_sha256"), length=64, field=f"datasets.{name}.sequence_list_sha256")

    baselines = config.get("baselines", {})
    for name in BASELINES:
        spec = baselines.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"missing frozen baseline entry: {name}")
        if spec.get("validated") is not True:
            raise ValueError(f"baseline {name} must be validated before freeze")
        _sha(spec.get("config_sha256"), length=64, field=f"baselines.{name}.config_sha256")
        _sha(
            spec.get("evidence_manifest_sha256"),
            length=64,
            field=f"baselines.{name}.evidence_manifest_sha256",
        )

    models = config.get("models", {})
    for name in MODELS:
        spec = models.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"missing frozen model artifact: {name}")
        _sha(spec.get("artifact_sha256"), length=64, field=f"models.{name}.artifact_sha256")
        _sha(spec.get("config_sha256"), length=64, field=f"models.{name}.config_sha256")

    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique values")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("every seed must be an integer")

    analysis = config.get("analysis", {})
    if analysis.get("multiplicity") != "holm":
        raise ValueError("analysis.multiplicity must remain 'holm'")
    if int(analysis.get("bootstrap_resamples", 0)) != 10000:
        raise ValueError("analysis.bootstrap_resamples must remain 10000")
    if float(analysis.get("h7_noninferiority_margin_pp", -1)) != 2.0:
        raise ValueError("analysis.h7_noninferiority_margin_pp must remain 2.0")
