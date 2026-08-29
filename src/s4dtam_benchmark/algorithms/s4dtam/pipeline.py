from __future__ import annotations

import hashlib
import json
import re
from time import perf_counter

import numpy as np

from s4dtam_benchmark.algorithms.base import AlgorithmAdapter
from s4dtam_benchmark.contracts import (
    MODALITIES,
    AlgorithmResult,
    AvailabilityState,
    RunContext,
    SequenceData,
)

from .encoders import (
    GNSSEncoder,
    IMUEncoder,
    LiDAREncoder,
    MaskedFusion,
    RGBEncoder,
    ThermalEncoder,
)
from .memory import LifecycleRules, ModalityNoiseModel, ResourceBudgets, TokenMemory
from .telemetry import EventLogConfig, JsonlEventLogger


class S4DTAMReference(AlgorithmAdapter):
    """Transparent CPU reference for the token lifecycle and sensor interfaces.

    Args:
        association_radius_m: Radial fallback association threshold in metres.
        encoder_dim: Common encoder feature dimension. The reference map requires three.
        encoder_scales: Optional per-modality input scaling factors.
        fusion_weights: Optional per-modality confidence multipliers.
        association_mode: Built-in primary association mode.
        association_rejection_threshold: Maximum accepted feature-association cost.
        lifecycle: Optional token lifecycle policy.
        budgets: Optional token, memory, history and update-time budgets.
        event_logging: Structured per-sequence event log configuration.

    Raises:
        ValueError: If ``encoder_dim`` is incompatible with XYZ token positions.
    """

    name = "s4d_tam_reference"

    def __init__(
        self,
        association_radius_m: float = 0.35,
        encoder_dim: int = 3,
        encoder_scales: dict[str, float] | None = None,
        fusion_weights: dict[str, float] | None = None,
        association_mode: str = "feature",
        association_rejection_threshold: float = 0.35,
        lifecycle: LifecycleRules | None = None,
        budgets: ResourceBudgets | None = None,
        event_logging: EventLogConfig | None = None,
        noise_model: ModalityNoiseModel | None = None,
    ) -> None:
        if encoder_dim != 3:
            raise ValueError("S4DTAMReference requires encoder_dim=3 for TokenMemory positions")
        self.association_radius_m = association_radius_m
        self.association_mode = association_mode
        self.association_rejection_threshold = association_rejection_threshold
        self.lifecycle = lifecycle
        self.budgets = budgets
        self.event_logging = event_logging or EventLogConfig()
        self.noise_model = noise_model or ModalityNoiseModel()
        self.calibration_parameters: dict[str, object] = {
            "feature_mean": [0.0, 0.0, 0.0], "feature_scale": [1.0, 1.0, 1.0],
            "covariance_scale": 1.0,
        }
        self.calibration_data_id = "uncalibrated"
        self.calibration_artifact: str | None = None
        scales = encoder_scales or {}
        encoder_types = {
            "rgb": RGBEncoder,
            "thermal": ThermalEncoder,
            "lidar": LiDAREncoder,
            "imu": IMUEncoder,
            "gnss": GNSSEncoder,
        }
        self.encoders = {
            name: kind(encoder_dim, scales.get(name, 1.0)) for name, kind in encoder_types.items()
        }
        self.fusion = MaskedFusion(encoder_dim, fusion_weights)

    def calibrate(
        self, sequences: list[SequenceData], context: RunContext, data_id: str
    ) -> dict[str, object]:
        """Fit uncertainty/OOD scaling only on an explicitly held-out split."""
        features, residual_sq = [], []
        for sequence in sequences:
            if sequence.observations is not None:
                values = np.asarray(sequence.observations, dtype=float)
                features.extend(values.tolist())
                residual_sq.extend(np.sum((values - sequence.gt_positions) ** 2, axis=1).tolist())
        array = np.asarray(features, dtype=float)
        if array.size == 0:
            raise ValueError("calibration split has no legacy observations")
        scale = np.std(array, axis=0).clip(min=1e-6)
        self.calibration_parameters = {
            "feature_mean": np.mean(array, axis=0).tolist(),
            "feature_scale": scale.tolist(),
            "covariance_scale": max(float(np.mean(residual_sq) / 3.0), 1e-6),
        }
        self.calibration_data_id = str(data_id)
        directory = context.output_dir / "model_artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}_calibration.json"
        payload = {"schema": "s4dtam-calibration/v1", "data_id": self.calibration_data_id,
                   "parameters": self.calibration_parameters}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.calibration_artifact = str(path.relative_to(context.output_dir))
        return payload

    def run(self, sequence: SequenceData, context: RunContext) -> AlgorithmResult:
        """Process one sequence and return trajectory, semantics and resource metrics.

        Args:
            sequence: Validated synchronized sensor or legacy observation sequence.
            context: Experiment output, seed and configuration context.

        Returns:
            Reference estimates and diagnostics in the benchmark result contract.

        Raises:
            ValueError: If the sequence has neither usable modalities nor valid
                three-dimensional legacy observations.
        """
        has_modalities = any(getattr(sequence, name) is not None for name in MODALITIES)
        reference_mode = not has_modalities
        if reference_mode and sequence.observations is None:
            raise ValueError("S4D-TAM requires a modality stream or legacy normalized observations")
        if reference_mode and np.shape(sequence.observations) != (len(sequence.timestamps), 3):
            raise ValueError("legacy normalized observations must have shape (samples, 3)")
        event_logger = None
        event_log_path = None
        if self.event_logging.enabled:
            dataset = _safe_path_component(sequence.dataset)
            sequence_id = _safe_path_component(sequence.sequence_id)
            event_log_path = (
                context.output_dir
                / self.event_logging.directory
                / dataset
                / f"{sequence_id}_{self.name}.jsonl"
            )
            event_logger = JsonlEventLogger(
                event_log_path,
                dataset=sequence.dataset,
                sequence=sequence.sequence_id,
                algorithm=self.name,
                flush_each_event=self.event_logging.flush_each_event,
            )
        memory = TokenMemory(
            self.association_radius_m,
            association_mode=self.association_mode,
            rejection_threshold=self.association_rejection_threshold,
            lifecycle=self.lifecycle,
            budgets=self.budgets,
            event_sink=event_logger,
            log_attention_components=self.event_logging.include_attention_components,
            noise_model=self.noise_model,
        )
        estimates, covariances, ood_scores, semantics, latency = [], [], [], [], []
        fused_states: list[int] = []
        last_observation = np.zeros(3)
        for index, timestamp in enumerate(sequence.timestamps):
            start = perf_counter()
            if reference_mode:
                observation = sequence.observations[index]
                quality = 1.0
                modality = "legacy"
                fused_states.append(int(AvailabilityState.AVAILABLE))
            else:
                states = {
                    name: int(sequence.availability_masks[name][index]) for name in MODALITIES
                }
                encoded = [
                    self.encoders[name].encode(getattr(sequence, name)[index], float(timestamp))
                    for name in MODALITIES
                    if getattr(sequence, name) is not None
                    and states[name] == AvailabilityState.AVAILABLE
                ]
                fused = self.fusion.fuse(encoded, states, float(timestamp))
                quality = float(np.mean([item.confidence for item in encoded])) if encoded else 0.0
                modality = encoded[0].modality if len(encoded) == 1 else "fused"
                fused_states.append(int(fused.state))
                observation = (
                    fused.features
                    if fused.state == AvailabilityState.AVAILABLE
                    else last_observation
                )
                if fused.state == AvailabilityState.AVAILABLE:
                    last_observation = observation
            semantic_hint = (
                int(sequence.semantic_observations[index])
                if sequence.semantic_observations is not None
                else None
            )
            token = memory.update(
                observation, float(timestamp), semantic_hint, modality=modality, quality=quality
            )
            estimates.append(token.position.copy())
            covariance_scale = float(self.calibration_parameters["covariance_scale"])
            covariances.append(token.covariance.copy() * covariance_scale)
            centre = np.asarray(self.calibration_parameters["feature_mean"], dtype=float)
            scale = np.asarray(self.calibration_parameters["feature_scale"], dtype=float)
            ood_scores.append(float(np.mean(((np.asarray(observation) - centre) / scale) ** 2)))
            semantics.append(int(np.argmax(token.semantic_logits)))
            latency.append((perf_counter() - start) * 1000.0)

        occupancy_pred = {}
        if sequence.occupancy_observations is not None:
            for horizon in sequence.occupancy_gt:
                steps = max(1, int(round(horizon / np.median(np.diff(sequence.timestamps)))))
                velocity_proxy = np.roll(sequence.occupancy_observations, steps, axis=0)
                velocity_proxy[:steps] = sequence.occupancy_observations[:steps]
                occupancy_pred[horizon] = np.clip(
                    0.65 * sequence.occupancy_observations + 0.25 * velocity_proxy + 0.05,
                    0.0,
                    1.0,
                )
        resource = {
            "token_count": float(len(memory.tokens)),
            "map_bytes": float(memory.map_bytes),
            "last_update_ms": float(memory.last_update_ms),
            "time_budget_exceeded": float(memory.time_budget_exceeded),
        }
        if memory.budgets.max_update_time_ms is not None:
            resource["update_time_budget_ms"] = float(memory.budgets.max_update_time_ms)
        if event_logger is not None:
            event_logger.emit(
                "run_completed",
                float(sequence.timestamps[-1]) if len(sequence.timestamps) else None,
                token_count=len(memory.tokens),
                map_bytes=memory.map_bytes,
                sample_count=len(sequence.timestamps),
            )
        return AlgorithmResult(
            algorithm=self.name,
            timestamps=sequence.timestamps,
            estimated_positions=np.asarray(estimates),
            pose_covariances=np.asarray(covariances),
            ood_scores=np.asarray(ood_scores),
            semantic_pred=np.asarray(semantics),
            occupancy_pred=occupancy_pred,
            latency_ms=np.asarray(latency),
            resource=resource,
            metadata={
                "implementation": "reference_cpu",
                "input_mode": "legacy_reference" if reference_mode else "multimodal_encoded",
                "fused_availability_states": fused_states,
                "not_flight_certified": True,
                "association": memory.association_summary,
                "calibration": {"data_id": self.calibration_data_id,
                                "artifact": self.calibration_artifact,
                                "parameters": self.calibration_parameters},
                "event_log": (
                    str(event_log_path.relative_to(context.output_dir))
                    if event_log_path is not None
                    else None
                ),
            },
        )


def _safe_path_component(value: str) -> str:
    """Convert an external identifier into a portable, traversal-safe file component."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned or 'unnamed'}-{digest}"
