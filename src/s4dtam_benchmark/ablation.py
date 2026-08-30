from __future__ import annotations

from typing import Any


EXPECTED_VARIANTS = {
    "H1": ("H1_no_semantics", "semantics"),
    "H2": ("H2_no_temporal_state", "temporal_state"),
    "H3": ("H3_no_calibrated_uncertainty", "calibrated_uncertainty"),
    "H4": ("H4_no_topology", "topology"),
    "H5": ("H5_no_reference_map", "reference_map"),
    "H6": ("H6_no_risk_prediction", "risk_prediction"),
    "H7": ("H7_no_token_lifecycle", "token_lifecycle"),
}


def validate_ablation_config(config: dict[str, Any]) -> None:
    """Reject incomplete, confounded, or test-contaminated ablation matrices."""
    errors: list[str] = []
    components = config.get("components")
    variants = config.get("variants")
    if not isinstance(components, dict) or not components:
        errors.append("components must be a non-empty mapping")
        components = {}
    if not isinstance(variants, list):
        errors.append("variants must be a list")
        variants = []

    by_hypothesis = {item.get("hypothesis"): item for item in variants if isinstance(item, dict)}
    names = [item.get("name") for item in variants if isinstance(item, dict)]
    if len(names) != len(set(names)):
        errors.append("variant names must be unique")
    control = by_hypothesis.get("control")
    if not control or control.get("name") != "full" or control.get("overrides") != {}:
        errors.append("matrix requires an unchanged 'full' control")

    for hypothesis, (expected_name, component) in EXPECTED_VARIANTS.items():
        variant = by_hypothesis.get(hypothesis)
        if variant is None:
            errors.append(f"missing variant for {hypothesis}")
            continue
        if variant.get("name") != expected_name:
            errors.append(f"{hypothesis} variant must be named {expected_name}")
        if variant.get("overrides") != {component: False}:
            errors.append(f"{expected_name} must disable only {component}")
        if components.get(component) is not True:
            errors.append(f"full control must enable {component}")

    allowed_hypotheses = {"control", *EXPECTED_VARIANTS}
    extras = set(by_hypothesis) - allowed_hypotheses
    if extras or len(variants) != len(allowed_hypotheses):
        errors.append("matrix must contain exactly full plus H1-H7")

    provenance = config.get("artifact_provenance", {})
    evaluation = set(config.get("evaluation_splits", []))
    artifacts: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        artifact = variant.get("artifact")
        if not isinstance(artifact, str) or not artifact:
            errors.append(f"{variant.get('name', '<unnamed>')} has no artifact")
            continue
        artifacts.append(artifact)
        record = provenance.get(artifact)
        if not isinstance(record, dict):
            errors.append(f"missing provenance for artifact {artifact}")
            continue
        trained = set(record.get("trained_on_splits", []))
        if not trained:
            errors.append(f"artifact {artifact} has no training splits")
        if trained & evaluation or "test" in trained:
            errors.append(f"artifact {artifact} was trained on an evaluation/test split")
        if not record.get("data_version"):
            errors.append(f"artifact {artifact} has no data version")
    if len(artifacts) != len(set(artifacts)):
        errors.append("variants must not share trained artifacts")

    if errors:
        raise ValueError("Invalid ablation configuration:\n- " + "\n- ".join(errors))
