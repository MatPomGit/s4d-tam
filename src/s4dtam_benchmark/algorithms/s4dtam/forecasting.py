"""Online, causal occupancy and motion forecasting.

The numerical reference forecaster intentionally has a small and auditable state.
Each call to :meth:`CausalForecaster.update` consumes exactly one timestamped map;
the implementation consequently cannot inspect frames after the prediction time.
Forecast targets are aligned in physical time, which gives defined behaviour for
irregular, delayed, and dropped samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class ForecastBatch:
    """Time-major probabilistic forecasts indexed by horizon in seconds.

    ``occupancy_probability`` parameterizes a Bernoulli distribution and its
    uncertainty is the corresponding variance ``p * (1 - p)``. ``flow_mean`` and
    ``flow_uncertainty`` parameterize an axis-independent Gaussian distribution;
    the latter contains its per-axis variance. Masks identify cells for which a
    target exists in time and the spatial projection remains inside the map.
    """

    occupancy_probability: dict[float, np.ndarray]
    flow_mean: dict[float, np.ndarray]
    occupancy_uncertainty: dict[float, np.ndarray]
    flow_uncertainty: dict[float, np.ndarray]
    observable_mask: dict[float, np.ndarray]
    target_indices: dict[float, np.ndarray]


@dataclass(frozen=True, slots=True)
class _FrameForecast:
    """Forecast components produced at one prediction time and horizon."""

    occupancy_probability: np.ndarray
    flow_mean: np.ndarray
    occupancy_variance: np.ndarray
    flow_variance: np.ndarray
    spatial_mask: np.ndarray


def target_indices(timestamps: np.ndarray, horizon_s: float) -> np.ndarray:
    """Find right-continuous forecast targets using physical timestamps.

    Args:
        timestamps: Strictly increasing, finite timestamps in seconds.
        horizon_s: Positive finite prediction horizon in seconds.

    Returns:
        For every source timestamp, the first sample index whose timestamp is at
        least ``source + horizon_s``. A value of ``-1`` means that the sequence
        ends before such a target exists.

    Raises:
        ValueError: If timestamps or the horizon violate the input contract.
    """
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("horizon_s must be positive and finite")
    indices = np.searchsorted(times, times + horizon_s, side="left").astype(int)
    indices[indices == len(times)] = -1
    return indices


class CausalForecaster:
    """Reference constant-velocity forecaster for occupancy grids.

    Motion evidence is the mass that disappeared and appeared between the two
    most recent frames. Unchanged occupancy is persisted, while newly appearing
    mass is extrapolated with constant velocity. This separates static occupancy
    from moving occupancy without using future frames.

    Args:
        horizons_s: Positive horizons for which every update produces a forecast.
        max_shift: Maximum absolute per-axis displacement inferred between two
            consecutive observations. It limits sensitivity to map noise.

    Raises:
        ValueError: If horizons or ``max_shift`` are invalid.
    """

    def __init__(self, horizons_s: Iterable[float], max_shift: int = 8) -> None:
        self.horizons = _normalize_horizons(horizons_s)
        if isinstance(max_shift, bool) or int(max_shift) != max_shift or max_shift < 0:
            raise ValueError("max_shift must be a non-negative integer")
        self.max_shift = int(max_shift)
        self._times: list[float] = []
        self._frames: list[np.ndarray] = []
        self._outputs: dict[float, list[_FrameForecast]] = {
            horizon: [] for horizon in self.horizons
        }

    def update(self, timestamp: float, occupancy: np.ndarray) -> None:
        """Consume one current observation and produce every configured horizon.

        Args:
            timestamp: Finite observation time, greater than the preceding time.
            occupancy: Grid of probabilities in ``[0, 1]``. Grid shape must remain
                constant throughout a sequence.

        Raises:
            ValueError: If the timestamp, values, or shape violate the contract.
        """
        frame = self._validate_observation(timestamp, occupancy)
        velocity, moving = self._estimate_motion(timestamp, frame)
        history_size = len(self._frames) + 1

        for horizon in self.horizons:
            displacement = np.rint(velocity * horizon).astype(int)
            shifted_moving, spatial_mask = _translate(moving, tuple(displacement))
            occupancy_mean = np.clip(frame - moving + shifted_moving, 0.0, 1.0)
            confidence = _forecast_confidence(history_size, horizon)
            probability = np.clip(0.5 + confidence * (occupancy_mean - 0.5), 1e-4, 1.0 - 1e-4)

            flow = np.zeros(frame.shape + (frame.ndim,), dtype=float)
            moving_support = shifted_moving > 0
            flow[moving_support] = displacement
            flow_variance = np.full_like(flow, (horizon / max(history_size - 1, 1)) ** 2)
            self._outputs[horizon].append(
                _FrameForecast(
                    occupancy_probability=probability,
                    flow_mean=flow,
                    occupancy_variance=probability * (1.0 - probability),
                    flow_variance=flow_variance,
                    spatial_mask=spatial_mask,
                )
            )

        self._times.append(float(timestamp))
        self._frames.append(frame.copy())

    def result(self) -> ForecastBatch:
        """Stack all forecasts without mutating or resetting forecaster state.

        Returns:
            A :class:`ForecastBatch` with one leading time dimension per array.

        Raises:
            RuntimeError: If no observation has been supplied yet.
        """
        if not self._times:
            raise RuntimeError("cannot build a forecast result before the first update")
        targets = {
            horizon: target_indices(np.asarray(self._times), horizon) for horizon in self.horizons
        }
        occupancy, flow, occupancy_variance, flow_variance, masks = {}, {}, {}, {}, {}
        for horizon, values in self._outputs.items():
            occupancy[horizon] = np.stack([value.occupancy_probability for value in values])
            flow[horizon] = np.stack([value.flow_mean for value in values])
            occupancy_variance[horizon] = np.stack([value.occupancy_variance for value in values])
            flow_variance[horizon] = np.stack([value.flow_variance for value in values])
            spatial = np.stack([value.spatial_mask for value in values])
            temporal_shape = (-1,) + (1,) * (spatial.ndim - 1)
            masks[horizon] = spatial & (targets[horizon] >= 0).reshape(temporal_shape)
        return ForecastBatch(occupancy, flow, occupancy_variance, flow_variance, masks, targets)

    def _validate_observation(self, timestamp: float, occupancy: np.ndarray) -> np.ndarray:
        """Validate and normalize a single occupancy observation."""
        if not np.isfinite(timestamp):
            raise ValueError("forecast timestamp must be finite")
        if self._times and timestamp <= self._times[-1]:
            raise ValueError("forecast timestamps must be strictly increasing")
        frame = np.asarray(occupancy, dtype=float)
        if frame.ndim < 1 or not np.all(np.isfinite(frame)):
            raise ValueError("occupancy frames must be finite, non-scalar arrays")
        if np.any((frame < 0.0) | (frame > 1.0)):
            raise ValueError("occupancy probabilities must lie in [0, 1]")
        if self._frames and frame.shape != self._frames[-1].shape:
            raise ValueError("occupancy frame shape changed within a sequence")
        return frame

    def _estimate_motion(
        self, timestamp: float, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate causal grid velocity and current moving occupancy mass."""
        velocity = np.zeros(frame.ndim, dtype=float)
        moving = np.zeros_like(frame)
        if not self._frames:
            return velocity, moving

        previous = self._frames[-1]
        appeared = np.clip(frame - previous, 0.0, 1.0)
        disappeared = np.clip(previous - frame, 0.0, 1.0)
        if np.sum(appeared) == 0 or np.sum(disappeared) == 0:
            return velocity, moving

        coordinates = np.indices(frame.shape).reshape(frame.ndim, -1)
        appeared_center = np.average(coordinates, axis=1, weights=appeared.ravel())
        disappeared_center = np.average(coordinates, axis=1, weights=disappeared.ravel())
        shift = np.clip(appeared_center - disappeared_center, -self.max_shift, self.max_shift)
        velocity = shift / (timestamp - self._times[-1])
        return velocity, appeared


def _normalize_horizons(horizons_s: Iterable[float]) -> tuple[float, ...]:
    """Return sorted unique finite horizons after validating configuration."""
    horizons = tuple(sorted({float(value) for value in horizons_s}))
    if not horizons or not np.all(np.isfinite(horizons)) or horizons[0] <= 0:
        raise ValueError("at least one positive finite forecast horizon is required")
    return horizons


def _forecast_confidence(history_size: int, horizon_s: float) -> float:
    """Compute bounded confidence that decays with horizon and sparse history."""
    history_confidence = min(0.95, 0.55 + 0.2 * min(history_size - 1, 2))
    return float(history_confidence * np.exp(-0.04 * horizon_s))


def _translate(array: np.ndarray, shift: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Translate an array without wraparound and return valid destination cells."""
    output = np.zeros_like(array, dtype=float)
    mask = np.zeros(array.shape, dtype=bool)
    source, destination = [], []
    for size, amount in zip(array.shape, shift, strict=True):
        if abs(amount) >= size:
            return output, mask
        source.append(slice(max(0, -amount), min(size, size - amount)))
        destination.append(slice(max(0, amount), min(size, size + amount)))
    output[tuple(destination)] = array[tuple(source)]
    mask[tuple(destination)] = True
    return output, mask
