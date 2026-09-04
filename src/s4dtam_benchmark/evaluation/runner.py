from __future__ import annotations

import numpy as np

from s4dtam_benchmark.contracts import AlgorithmResult, SequenceData

from .efficiency import efficiency_metrics
from .forecast import flow_metrics, occupancy_metrics
from .navigation import navigation_metrics
from .semantic import semantic_metrics
from .trajectory import rotation_rpe_deg, trajectory_metrics
from .uncertainty import (
    binary_risk_metrics,
    ood_metrics,
    pose_calibration_metrics,
    pose_uncertainty_metrics,
    selective_risk_metrics,
)

# Absolute tolerance for comparing dataset and result timestamps, in seconds.
TIMESTAMP_ATOL_SECONDS = 1e-6


def _matching_pair(
    name: str, target: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target_array, prediction_array = np.asarray(target), np.asarray(prediction)
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            f"{name} target and prediction shapes must match exactly: "
            f"{target_array.shape} != {prediction_array.shape}"
        )
    if target_array.size == 0:
        raise ValueError(f"{name} target and prediction must not be empty")
    return target_array, prediction_array


def _masked_forecast_pair(
    name: str,
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    target_array, prediction_array = _matching_pair(name, target, prediction)
    if mask is None:
        return target_array, prediction_array
    mask_array = np.asarray(mask)
    expected_shape = target_array.shape if name == "occupancy" else target_array.shape[:-1]
    if mask_array.dtype != np.bool_ or mask_array.shape != expected_shape:
        raise ValueError(
            f"{name} forecast mask must be boolean with shape {expected_shape}, "
            f"got dtype={mask_array.dtype}, shape={mask_array.shape}"
        )
    masked_target, masked_prediction = target_array[mask_array], prediction_array[mask_array]
    return _matching_pair(f"masked {name}", masked_target, masked_prediction)


def validate_time_contract(
    sequence: SequenceData,
    result: AlgorithmResult,
    *,
    timestamp_atol_seconds: float = TIMESTAMP_ATOL_SECONDS,
) -> None:
    """Require a one-to-one timestamp match; no interpolation is performed."""
    expected = sequence.timestamps
    actual = result.timestamps
    overlap = min(len(expected), len(actual))
    max_difference = (
        float(np.max(np.abs(expected[:overlap] - actual[:overlap]))) if overlap else float("inf")
    )
    same_count = len(expected) == len(actual)
    timestamps_match = same_count and np.allclose(
        expected, actual, rtol=0.0, atol=timestamp_atol_seconds
    )
    if not timestamps_match:
        raise ValueError(
            "timestamp contract mismatch: "
            f"dataset={sequence.dataset!r}, sequence_id={sequence.sequence_id!r}, "
            f"algorithm={result.algorithm!r}, dataset_samples={len(expected)}, "
            f"result_samples={len(actual)}, max_time_difference_s={max_difference:g}"
        )


def evaluate_result(
    sequence: SequenceData, result: AlgorithmResult
) -> tuple[dict[str, float], list[str]]:
    validate_time_contract(sequence, result)
    metrics = trajectory_metrics(sequence.gt_positions, result.estimated_positions)
    unavailable: list[str] = []
    metrics.update(
        efficiency_metrics(result.latency_ms, result.resource, result.planner_cost_diagnostics)
    )
    if sequence.gt_quaternions is not None and result.estimated_quaternions is not None:
        quaternion_target, quaternion_prediction = _matching_pair(
            "quaternion", sequence.gt_quaternions, result.estimated_quaternions
        )
        metrics["trajectory/rpe_rotation_rmse_deg"] = rotation_rpe_deg(
            quaternion_target, quaternion_prediction
        )
    else:
        unavailable.append("trajectory rotation: quaternion ground truth or prediction absent")

    if result.pose_covariances is not None:
        metrics.update(
            pose_uncertainty_metrics(
                sequence.gt_positions, result.estimated_positions, result.pose_covariances
            )
        )
        metrics.update(
            pose_calibration_metrics(
                sequence.gt_positions, result.estimated_positions, result.pose_covariances
            )
        )
    else:
        unavailable.append("pose uncertainty: covariance prediction absent")

    selection_uncertainty = None
    if result.pose_covariances is not None:
        selection_uncertainty = np.trace(result.pose_covariances, axis1=1, axis2=2)
    elif result.ood_scores is not None:
        selection_uncertainty = result.ood_scores
    if selection_uncertainty is not None:
        metrics.update(
            selective_risk_metrics(
                sequence.gt_positions, result.estimated_positions, selection_uncertainty
            )
        )
    else:
        unavailable.append("selective risk: uncertainty prediction absent")

    if result.ood_scores is not None:
        labels = sequence.metadata.get("ood_labels")
        if labels is not None:
            labels, scores = _matching_pair("OOD", labels, result.ood_scores)
            metrics.update(ood_metrics(labels, scores))
        else:
            unavailable.append("OOD: binary labels absent")
    else:
        unavailable.append("OOD: score prediction absent")

    if sequence.semantic_gt is not None and result.semantic_pred is not None:
        target, prediction = _matching_pair("semantic", sequence.semantic_gt, result.semantic_pred)
        metrics.update(semantic_metrics(target, prediction))
    else:
        unavailable.append("semantic: ground truth or prediction absent")

    for horizon, target in sequence.occupancy_gt.items():
        if horizon in result.occupancy_pred:
            try:
                target, prediction = _masked_forecast_pair(
                    "occupancy",
                    target,
                    result.occupancy_pred[horizon],
                    result.forecast_observable_mask.get(horizon),
                )
            except ValueError as error:
                if "must not be empty" not in str(error):
                    raise
                unavailable.append(f"forecast/{horizon:g}s: no observable targets")
                continue
            for key, value in occupancy_metrics(target, prediction).items():
                metrics[f"forecast/{horizon:g}s/{key}"] = value
        else:
            unavailable.append(f"forecast/{horizon:g}s: prediction absent")

    for horizon, target in sequence.flow_gt.items():
        if horizon in result.flow_pred:
            try:
                target, prediction = _masked_forecast_pair(
                    "flow",
                    target,
                    result.flow_pred[horizon],
                    result.forecast_observable_mask.get(horizon),
                )
            except ValueError as error:
                if "must not be empty" not in str(error):
                    raise
                unavailable.append(f"flow/{horizon:g}s: no observable targets")
                continue
            for key, value in flow_metrics(target, prediction).items():
                metrics[f"flow/{horizon:g}s/{key}"] = value
        else:
            unavailable.append(f"flow/{horizon:g}s: prediction absent")

    if result.navigation or result.planned_trajectory is not None:
        metrics.update(
            navigation_metrics(
                result.navigation,
                sequence.navigation_gt,
                result.planned_trajectory,
                result.planner_cost_diagnostics,
            )
        )
    else:
        unavailable.append("navigation: closed-loop trace absent")
    if sequence.risk_gt is not None and result.risk_pred is not None:
        risk_target, risk_prediction = _matching_pair("risk", sequence.risk_gt, result.risk_pred)
        metrics.update(binary_risk_metrics(risk_target, risk_prediction))
    else:
        unavailable.append("risk: ground truth or probability prediction absent")
    return metrics, unavailable
