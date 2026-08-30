from __future__ import annotations

from typing import Any

from s4dtam_benchmark.ablation import validate_ablation_config

COMPARISON_LEVELS = frozenset({"external", "internal"})
CORE_EXTERNAL_BASELINES = frozenset({"orb_slam3", "vins_mono", "fast_lio2", "lio_sam"})


def validate_comparison_config(config: dict[str, Any]) -> None:
    """Validate the boundary between external benchmarking and mechanism studies.

    External comparison answers whether S4D-TAM performs competitively against
    independently implemented navigation systems. Internal comparison answers
    which S4D-TAM components contribute to the full model. Mixing these questions
    in one matrix would confound algorithm identity with component ablation, so the
    two levels intentionally use different configuration contracts.
    """
    level = config.get("comparison_level")
    if level not in COMPARISON_LEVELS:
        raise ValueError("comparison_level must be either 'external' or 'internal'")
    if level == "external":
        _validate_external(config)
        return
    _validate_internal(config)


def _validate_external(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if "variants" in config or "components" in config:
        errors.append("external comparison must not contain internal ablation variants/components")

    algorithms = config.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        errors.append("external comparison requires a non-empty algorithms list")
        algorithms = []

    candidate_name = config.get("candidate_algorithm")
    if not isinstance(candidate_name, str) or not candidate_name:
        errors.append("external comparison requires candidate_algorithm")

    names: list[str] = []
    candidate_specs: list[dict[str, Any]] = []
    baseline_names: set[str] = set()
    for spec in algorithms:
        if not isinstance(spec, dict):
            errors.append("every external algorithm entry must be a mapping")
            continue
        name = spec.get("name")
        role = spec.get("role")
        if not isinstance(name, str) or not name:
            errors.append("every external algorithm entry requires a name")
            continue
        names.append(name)
        if role == "candidate":
            candidate_specs.append(spec)
        elif role == "baseline":
            baseline_names.add(name)
            if spec.get("type") != "external_artifact":
                errors.append(f"external baseline {name} must use type external_artifact")
        else:
            errors.append(f"algorithm {name} must declare role candidate or baseline")

    if len(names) != len(set(names)):
        errors.append("external algorithm names must be unique")
    if len(candidate_specs) != 1:
        errors.append("external comparison requires exactly one candidate algorithm")
    elif candidate_specs[0].get("name") != candidate_name:
        errors.append("candidate_algorithm must match the algorithm whose role is candidate")

    missing = CORE_EXTERNAL_BASELINES - baseline_names
    if missing:
        errors.append("external comparison is missing core baselines: " + ", ".join(sorted(missing)))

    if errors:
        raise ValueError("Invalid external comparison configuration:\n- " + "\n- ".join(errors))


def _validate_internal(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if "algorithms" in config or "candidate_algorithm" in config:
        errors.append("internal comparison must use full vs H1-H7, not an external algorithm list")
    if errors:
        raise ValueError("Invalid internal comparison configuration:\n- " + "\n- ".join(errors))
    validate_ablation_config(config)
