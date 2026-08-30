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


def evaluate_result(
    sequence: SequenceData, result: AlgorithmResult
) -> tuple[dict[str, float], list[str]]:
    metrics = trajectory_metrics(sequence.gt_positions, result.estimated_positions)
    unavailable: list[str] = []
    metrics.update(
        efficiency_metrics(result.latency_ms, result.resource, result.planner_cost_diagnostics)
    )
    if sequence.gt_quaternions is not None and result.estimated_quaternions is not None:
        metrics["trajectory/rpe_rotation_rmse_deg"] = rotation_rpe_deg(
            sequence.gt_quaternions, result.estimated_quaternions
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
            metrics.update(ood_metrics(np.asarray(labels), result.ood_scores))
        else:
            unavailable.append("OOD: binary labels absent")
    else:
        unavailable.append("OOD: score prediction absent")

    if sequence.semantic_gt is not None and result.semantic_pred is not None:
        metrics.update(semantic_metrics(sequence.semantic_gt, result.semantic_pred))
    else:
        unavailable.append("semantic: ground truth or prediction absent")

    for horizon, target in sequence.occupancy_gt.items():
        if horizon in result.occupancy_pred:
            prediction = result.occupancy_pred[horizon]
            mask = result.forecast_observable_mask.get(horizon)
            if mask is not None:
                target, prediction = np.asarray(target)[mask], prediction[mask]
            if prediction.size == 0:
                unavailable.append(f"forecast/{horizon:g}s: no observable targets")
                continue
            for key, value in occupancy_metrics(target, prediction).items():
                metrics[f"forecast/{horizon:g}s/{key}"] = value
        else:
            unavailable.append(f"forecast/{horizon:g}s: prediction absent")

    for horizon, target in sequence.flow_gt.items():
        if horizon in result.flow_pred:
            prediction = result.flow_pred[horizon]
            mask = result.forecast_observable_mask.get(horizon)
            if mask is not None:
                target, prediction = np.asarray(target)[mask], prediction[mask]
            if prediction.size == 0:
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
        metrics.update(binary_risk_metrics(sequence.risk_gt, result.risk_pred))
    else:
        unavailable.append("risk: ground truth or probability prediction absent")
    return metrics, unavailable
