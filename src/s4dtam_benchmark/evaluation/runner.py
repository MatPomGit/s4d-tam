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


def _tracking_mask(result: AlgorithmResult) -> np.ndarray:
    raw = result.metadata.get("tracking_valid")
    if raw is None:
        return np.ones(len(result.timestamps), dtype=np.bool_)
    mask = np.asarray(raw)
    if mask.dtype != np.bool_ or mask.shape != (len(result.timestamps),):
        raise ValueError("result metadata tracking_valid must be boolean with shape [N]")
    return mask


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _failed_trajectory_metrics() -> dict[str, float]:
    return {
        "trajectory/ate_rmse_m": float("nan"),
        "trajectory/ate_median_m": float("nan"),
        "trajectory/ate_p95_m": float("nan"),
        "trajectory/rpe_translation_rmse_m": float("nan"),
        "trajectory/final_drift_m": float("nan"),
        "trajectory/final_drift_percent": float("nan"),
        "trajectory/path_length_m": float("nan"),
        "trajectory/alignment_scale": float("nan"),
    }


def evaluate_result(
    sequence: SequenceData, result: AlgorithmResult
) -> tuple[dict[str, float], list[str]]:
    validate_time_contract(sequence, result)
    valid = _tracking_mask(result)
    alignment_mode = str(result.metadata.get("alignment_mode", "se3"))
    unavailable: list[str] = []

    metrics: dict[str, float] = {
        "tracking/valid_fraction": float(np.mean(valid)),
        "tracking/failure_rate": float(1.0 - np.mean(valid)),
        "tracking/valid_samples": float(np.count_nonzero(valid)),
        "tracking/total_samples": float(len(valid)),
        "tracking/longest_failure_run_frames": float(_longest_false_run(valid)),
        "tracking/final_frame_valid": float(valid[-1]),
    }
    if np.count_nonzero(valid) >= 2:
        try:
            metrics.update(
                trajectory_metrics(
                    sequence.gt_positions[valid],
                    result.estimated_positions[valid],
                    alignment_mode=alignment_mode,
                )
            )
        except ValueError as error:
            metrics.update(_failed_trajectory_metrics())
            unavailable.append(f"trajectory accuracy: {error}")
    else:
        metrics.update(_failed_trajectory_metrics())
        unavailable.append("trajectory accuracy: fewer than two valid tracking samples")

    metrics.update(
        efficiency_metrics(result.latency_ms, result.resource, result.planner_cost_diagnostics)
    )
    if sequence.gt_quaternions is not None and result.estimated_quaternions is not None:
        quaternion_target, quaternion_prediction = _matching_pair(
            "quaternion", sequence.gt_quaternions, result.estimated_quaternions
        )
        metrics["trajectory/rpe_rotation_rmse_deg"] = rotation_rpe_deg(
            quaternion_target, quaternion_prediction, valid_mask=valid
        )
    else:
        unavailable.append("trajectory rotation: quaternion ground truth or prediction absent")

    if result.pose_covariances is not None:
        metrics.update(
            pose_uncertainty_metrics(
                sequence.gt_positions[valid],
                result.estimated_positions[valid],
                result.pose_covariances[valid],
            )
        )
        metrics.update(
            pose_calibration_metrics(
                sequence.gt_positions[valid],
                result.estimated_positions[valid],
                result.pose_covariances[valid],
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
