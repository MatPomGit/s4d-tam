from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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
from .calibration import CalibrationParameters, fit_calibration
from .memory import LifecycleRules, ModalityNoiseModel, ResourceBudgets, TokenMemory
from .telemetry import EventLogConfig, JsonlEventLogger
from .reference_map import ReferenceMap
from .topology import TopologicalGraph


@dataclass(frozen=True, slots=True)
class _FrameMeasurement:
    """Validated three-dimensional measurement consumed by token memory."""

    position: np.ndarray
    modality: str
    quality: float
    availability: AvailabilityState


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
        noise_model: Optional modality-, quality-, and time-dependent noise model.

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
        reference_map: ReferenceMap | None = None,
        map_enabled: bool = True,
        topology: TopologicalGraph | None = None,
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
        self.map_enabled = bool(map_enabled)
        if topology is not None and reference_map is not None:
            if topology.reference_map is not reference_map:
                raise ValueError("topology and reference_map must refer to the same map object")
        self.reference_map = reference_map or (
            topology.reference_map if topology is not None else None
        )
        self.topology = None
        if self.map_enabled:
            self.topology = topology or (
                TopologicalGraph(reference_map) if reference_map is not None else None
            )
        self.calibration_parameters = CalibrationParameters()
        self.calibration_data_id = "uncalibrated"
        self.calibration_artifact: str | None = None
        scales = encoder_scales or {}
        self.encoders = {
            "rgb": RGBEncoder(encoder_dim, scales.get("rgb", 1.0)),
            "thermal": ThermalEncoder(encoder_dim, scales.get("thermal", 1.0)),
            "lidar": LiDAREncoder(encoder_dim, scales.get("lidar", 1.0)),
            "imu": IMUEncoder(encoder_dim, scales.get("imu", 1.0)),
            "gnss": GNSSEncoder(encoder_dim, scales.get("gnss", 1.0)),
        }
        self.fusion = MaskedFusion(encoder_dim, fusion_weights)

    def calibrate(
        self, sequences: list[SequenceData], context: RunContext, data_id: str
    ) -> dict[str, object]:
        """Fit and persist calibration parameters on an explicitly held-out split.

        Args:
            sequences: Calibration-only sequences, never test sequences.
            context: Run context defining the artifact output directory.
            data_id: Stable identifier of the exact calibration data selection.

        Returns:
            JSON-compatible calibration artifact payload.

        Raises:
            ValueError: If ``data_id`` is empty or no usable calibration samples exist.
        """
        if not data_id.strip():
            raise ValueError("calibration data_id must be non-empty")
        features: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        covariances: list[np.ndarray] = []
        for sequence in sequences:
            self._validate_input(sequence)
            previous_timestamp: float | None = None
            last_position = np.zeros(3, dtype=float)
            for index, timestamp in enumerate(sequence.timestamps):
                measurement = self._measurement_at(sequence, index, last_position)
                if measurement.availability != AvailabilityState.AVAILABLE:
                    continue
                dt = 0.0 if previous_timestamp is None else float(timestamp - previous_timestamp)
                features.append(measurement.position)
                errors.append(measurement.position - sequence.gt_positions[index])
                covariances.append(
                    self.noise_model.covariance(measurement.modality, measurement.quality, dt)
                )
                last_position = measurement.position
                previous_timestamp = float(timestamp)
        self.calibration_parameters = fit_calibration(
            np.asarray(features), np.asarray(errors), np.asarray(covariances)
        )
        self.calibration_data_id = str(data_id)
        directory = context.output_dir / "model_artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        data_digest = hashlib.sha256(data_id.encode("utf-8")).hexdigest()[:12]
        path = directory / f"{self.name}_calibration_{data_digest}.json"
        payload: dict[str, object] = {
            "schema": "s4dtam-calibration/v1",
            "data_id": self.calibration_data_id,
            "parameters": self.calibration_parameters.to_dict(),
        }
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_path.replace(path)
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
        reference_mode = self._validate_input(sequence)
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
        map_confidence: list[float] = []
        accepted_matches: list[dict[str, object]] = []
        rejected_matches: list[dict[str, object]] = []
        relocalizations: list[int] = []
        tracking_lost = False
        last_observation = np.zeros(3)
        for index, timestamp in enumerate(sequence.timestamps):
            start = perf_counter()
            measurement = self._measurement_at(sequence, index, last_observation)
            observation = measurement.position
            fused_states.append(int(measurement.availability))
            if measurement.availability == AvailabilityState.AVAILABLE:
                last_observation = observation
            else:
                tracking_lost = True
            semantic_hint = (
                int(sequence.semantic_observations[index])
                if sequence.semantic_observations is not None
                else None
            )
            token = memory.update(
                observation,
                float(timestamp),
                semantic_hint,
                modality=measurement.modality,
                quality=measurement.quality,
            )
            estimate = token.position.copy()
            confidence = 0.0
            if self.topology is not None and measurement.availability == AvailabilityState.AVAILABLE:
                descriptor = self._map_descriptor(sequence, index, observation)
                geometry_position = self._map_position(sequence, index, observation)
                # Retrieval is descriptor-only; geometry is deliberately evaluated
                # against the current sensor pose rather than token-memory history.
                match, candidates, rejected = self.topology.match(descriptor, geometry_position)
                rejected_matches.extend(
                    {"sample": index, **item} for item in rejected
                )
                if match is not None:
                    estimate = geometry_position + match.correction
                    confidence = match.confidence
                    accepted_matches.append(
                        {"sample": index, "token_id": match.token_id,
                         "confidence": match.confidence, "residual_m": match.residual_m}
                    )
                    if tracking_lost:
                        relocalizations.append(index)
                    tracking_lost = False
                elif candidates:
                    tracking_lost = True
            map_confidence.append(confidence)
            estimates.append(estimate)
            covariance_scale = self.calibration_parameters.covariance_scale
            covariances.append(token.covariance.copy() * covariance_scale)
            ood_scores.append(self.calibration_parameters.ood_score(observation))
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
                "map_correction": {
                    "enabled": self.topology is not None,
                    "mode": "reference_map" if self.topology is not None else "mapless",
                    "confidence": map_confidence,
                    "accepted_matches": accepted_matches,
                    "rejected_matches": rejected_matches,
                    "relocalizations": relocalizations,
                    "relocalized": bool(relocalizations),
                },
                "calibration": {
                    "data_id": self.calibration_data_id,
                    "artifact": self.calibration_artifact,
                    "parameters": self.calibration_parameters.to_dict(),
                },
                "event_log": (
                    str(event_log_path.relative_to(context.output_dir))
                    if event_log_path is not None
                    else None
                ),
            },
        )

    def _map_descriptor(
        self, sequence: SequenceData, index: int, observation: np.ndarray
    ) -> np.ndarray:
        """Return an explicit descriptor when supplied, otherwise use encoded XYZ."""
        descriptors = sequence.metadata.get("map_descriptors")
        if descriptors is not None:
            values = np.asarray(descriptors, dtype=float)
            if (
                values.ndim != 2
                or values.shape[0] != len(sequence.timestamps)
                or values.shape[1] == 0
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    "metadata.map_descriptors must contain one non-empty finite vector per sample"
                )
            return values[index]
        return np.asarray(observation, dtype=float)

    def _map_position(
        self, sequence: SequenceData, index: int, observation: np.ndarray
    ) -> np.ndarray:
        """Return a pose in the map frame when the dataset exposes one explicitly."""
        positions = sequence.metadata.get("map_positions")
        if positions is None:
            return np.asarray(observation, dtype=float)
        values = np.asarray(positions, dtype=float)
        if values.shape != (len(sequence.timestamps), 3) or not np.all(np.isfinite(values)):
            raise ValueError("metadata.map_positions must be a finite [samples, 3] array")
        position = values[index]
        frame = sequence.metadata.get("map_position_frame", "map")
        if frame == "map":
            return position
        if self.reference_map is None or not isinstance(frame, str):
            raise ValueError("map_position_frame requires a reference map and a string frame name")
        return self.reference_map.transform(position, frame, "map")

    def _validate_input(self, sequence: SequenceData) -> bool:
        """Validate algorithm-specific inputs and return whether legacy mode is active."""
        reference_mode = not any(getattr(sequence, name) is not None for name in MODALITIES)
        if reference_mode:
            observations = sequence.observations
            if observations is None:
                raise ValueError(
                    "S4D-TAM requires a modality stream or legacy normalized observations"
                )
            if observations.shape != (len(sequence.timestamps), 3):
                raise ValueError("legacy normalized observations must have shape (samples, 3)")
        return reference_mode

    def _measurement_at(
        self, sequence: SequenceData, index: int, last_position: np.ndarray
    ) -> _FrameMeasurement:
        """Build one modality-aware measurement for a synchronized frame.

        Args:
            sequence: Source sequence containing legacy or multimodal observations.
            index: Zero-based sample index.
            last_position: Most recent available position used during outages.

        Returns:
            Position, source modality, quality, and availability state.
        """
        if not any(getattr(sequence, name) is not None for name in MODALITIES):
            assert sequence.observations is not None
            return _FrameMeasurement(
                np.asarray(sequence.observations[index], dtype=float),
                "legacy",
                1.0,
                AvailabilityState.AVAILABLE,
            )
        states = {name: int(sequence.availability_masks[name][index]) for name in MODALITIES}
        timestamp = float(sequence.timestamps[index])
        encoded = [
            self.encoders[name].encode(getattr(sequence, name)[index], timestamp)
            for name in MODALITIES
            if getattr(sequence, name) is not None and states[name] == AvailabilityState.AVAILABLE
        ]
        fused = self.fusion.fuse(encoded, states, timestamp)
        if fused.state != AvailabilityState.AVAILABLE:
            return _FrameMeasurement(
                np.asarray(last_position, dtype=float).copy(), "fused", 0.0, fused.state
            )
        quality = float(np.mean([item.confidence for item in encoded]))
        modality = encoded[0].modality if len(encoded) == 1 else "fused"
        return _FrameMeasurement(fused.features.copy(), modality, quality, fused.state)


def _safe_path_component(value: str) -> str:
    """Convert an external identifier into a portable, traversal-safe file component."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned or 'unnamed'}-{digest}"
