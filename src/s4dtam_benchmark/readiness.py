from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MANDATORY_DATASETS = ("tartanair", "blackbird", "marsim", "aeroverse")
MANDATORY_BASELINES = ("orb_slam3", "vins_mono", "fast_lio2", "lio_sam")
VALID_PAIR_STATES = {"supported", "blocked", "not_applicable"}
VALID_METRIC_STATES = {"supported", "derived", "blocked", "unavailable"}


@dataclass(frozen=True, slots=True)
class ReadinessSummary:
    supported_pairs: int
    blocked_pairs: int
    not_applicable_pairs: int
    publication_ready: bool


def validate_readiness_matrix(config: dict[str, Any]) -> ReadinessSummary:
    """Validate the frozen dataset/sensor/baseline/metric capability matrix.

    The matrix is deliberately explicit. A baseline can be marked ``supported`` only
    when every required sensor is declared by the dataset. ``blocked`` means that the
    pairing is scientifically meaningful but some reproducibility gate is still open.
    ``not_applicable`` means the sensing assumptions do not match and the pair must not
    be included in the external comparison.
    """
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("readiness schema_version must be 1")

    datasets = config.get("datasets")
    baselines = config.get("baselines")
    matrix = config.get("matrix")
    if not isinstance(datasets, dict) or not isinstance(baselines, dict) or not isinstance(matrix, dict):
        raise ValueError("readiness matrix requires datasets, baselines and matrix mappings")

    missing_datasets = sorted(set(MANDATORY_DATASETS) - set(datasets))
    missing_baselines = sorted(set(MANDATORY_BASELINES) - set(baselines))
    if missing_datasets:
        raise ValueError(f"missing mandatory datasets: {', '.join(missing_datasets)}")
    if missing_baselines:
        raise ValueError(f"missing mandatory baselines: {', '.join(missing_baselines)}")

    baseline_requirements: dict[str, set[str]] = {}
    for name, spec in baselines.items():
        required = set(spec.get("required_sensors", []))
        if not required:
            raise ValueError(f"baseline {name} must declare required_sensors")
        baseline_requirements[name] = required
        if not str(spec.get("upstream", "")).strip():
            raise ValueError(f"baseline {name} must declare upstream")

    supported = blocked = not_applicable = 0
    for dataset_name in MANDATORY_DATASETS:
        dataset = datasets[dataset_name]
        sensors = set(dataset.get("sensors", []))
        if not sensors:
            raise ValueError(f"dataset {dataset_name} must declare sensors")
        metric_families = dataset.get("metrics", {})
        if not isinstance(metric_families, dict) or not metric_families:
            raise ValueError(f"dataset {dataset_name} must declare metric availability")
        for metric, state in metric_families.items():
            if state not in VALID_METRIC_STATES:
                raise ValueError(f"dataset {dataset_name} metric {metric} has invalid state {state!r}")

        row = matrix.get(dataset_name)
        if not isinstance(row, dict):
            raise ValueError(f"matrix is missing row for {dataset_name}")
        for baseline_name in MANDATORY_BASELINES:
            pair = row.get(baseline_name)
            if not isinstance(pair, dict):
                raise ValueError(f"matrix is missing pair {dataset_name}/{baseline_name}")
            state = pair.get("state")
            if state not in VALID_PAIR_STATES:
                raise ValueError(f"invalid state for {dataset_name}/{baseline_name}: {state!r}")
            reason = str(pair.get("reason", "")).strip()
            if state != "supported" and not reason:
                raise ValueError(f"{dataset_name}/{baseline_name} {state} requires a reason")

            requirements_met = baseline_requirements[baseline_name] <= sensors
            if state == "supported" and not requirements_met:
                missing = sorted(baseline_requirements[baseline_name] - sensors)
                raise ValueError(
                    f"{dataset_name}/{baseline_name} cannot be supported; missing sensors: {', '.join(missing)}"
                )
            if state == "not_applicable" and requirements_met:
                raise ValueError(
                    f"{dataset_name}/{baseline_name} is marked not_applicable although required sensors exist"
                )
            if state == "supported":
                supported += 1
            elif state == "blocked":
                blocked += 1
            else:
                not_applicable += 1

    publication_ready = blocked == 0 and all(
        bool(datasets[name].get("release_frozen")) for name in MANDATORY_DATASETS
    )
    return ReadinessSummary(supported, blocked, not_applicable, publication_ready)


def render_readiness_summary(config: dict[str, Any]) -> str:
    summary = validate_readiness_matrix(config)
    return (
        f"supported_pairs={summary.supported_pairs} "
        f"blocked_pairs={summary.blocked_pairs} "
        f"not_applicable_pairs={summary.not_applicable_pairs} "
        f"publication_ready={str(summary.publication_ready).lower()}"
    )
