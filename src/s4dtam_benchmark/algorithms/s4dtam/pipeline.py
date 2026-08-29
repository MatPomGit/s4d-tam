from __future__ import annotations

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
from .memory import TokenMemory


class S4DTAMReference(AlgorithmAdapter):
    """Transparent CPU reference for the proposed token lifecycle and interfaces."""

    name = "s4d_tam_reference"

    def __init__(
        self,
        association_radius_m: float = 0.35,
        encoder_dim: int = 3,
        encoder_scales: dict[str, float] | None = None,
        fusion_weights: dict[str, float] | None = None,
    ):
        if encoder_dim != 3:
            raise ValueError("S4DTAMReference requires encoder_dim=3 for TokenMemory positions")
        self.association_radius_m = association_radius_m
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

    def run(self, sequence: SequenceData, context: RunContext) -> AlgorithmResult:
        has_modalities = any(getattr(sequence, name) is not None for name in MODALITIES)
        reference_mode = not has_modalities
        if reference_mode and sequence.observations is None:
            raise ValueError("S4D-TAM requires a modality stream or legacy normalized observations")
        if reference_mode and np.shape(sequence.observations) != (len(sequence.timestamps), 3):
            raise ValueError("legacy normalized observations must have shape (samples, 3)")
        memory = TokenMemory(self.association_radius_m)
        estimates, semantics, latency = [], [], []
        fused_states: list[int] = []
        last_observation = np.zeros(3)
        for index, timestamp in enumerate(sequence.timestamps):
            start = perf_counter()
            if reference_mode:
                observation = sequence.observations[index]
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
            token = memory.update(observation, float(timestamp), semantic_hint)
            estimates.append(token.position.copy())
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
        return AlgorithmResult(
            algorithm=self.name,
            timestamps=sequence.timestamps,
            estimated_positions=np.asarray(estimates),
            semantic_pred=np.asarray(semantics),
            occupancy_pred=occupancy_pred,
            latency_ms=np.asarray(latency),
            resource={
                "token_count": float(len(memory.tokens)),
                "map_bytes": float(
                    sum(
                        token.position.nbytes
                        + token.covariance.nbytes
                        + token.velocity.nbytes
                        + token.semantic_logits.nbytes
                        for token in memory.tokens
                    )
                ),
            },
            metadata={
                "implementation": "reference_cpu",
                "input_mode": "legacy_reference" if reference_mode else "multimodal_encoded",
                "fused_availability_states": fused_states,
                "not_flight_certified": True,
            },
        )
