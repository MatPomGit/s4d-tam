"""Deterministic, uncertainty-aware local trajectory planning.

The module keeps the four inputs to planning explicit: :class:`PlannerState`,
:class:`PlannerGoal`, :class:`DynamicsConstraints`, and :class:`PredictiveMap`.
It is a small, auditable CPU reference rather than a flight-control component.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Any

import numpy as np

_COST_NAMES = ("collision_risk", "energy", "time", "goal_progress", "information_value")
_EPSILON = 1e-12


def _xyz(value: np.ndarray, name: str) -> np.ndarray:
    """Return an owned, finite XYZ vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite XYZ vector")
    return array.copy()


@dataclass(frozen=True, slots=True)
class PlannerState:
    """Vehicle kinematic and resource state at one planning instant."""

    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time_s: float = 0.0
    energy_wh: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _xyz(self.position, "position"))
        object.__setattr__(self, "velocity", _xyz(self.velocity, "velocity"))
        if not np.isfinite(self.time_s) or not np.isfinite(self.energy_wh):
            raise ValueError("time and consumed energy must be finite")
        if self.time_s < 0 or self.energy_wh < 0:
            raise ValueError("time and consumed energy must be non-negative")


@dataclass(frozen=True, slots=True)
class PlannerGoal:
    """Target position and the radius within which it is considered reached."""

    position: np.ndarray
    tolerance_m: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _xyz(self.position, "goal position"))
        if not np.isfinite(self.tolerance_m) or self.tolerance_m <= 0:
            raise ValueError("goal tolerance must be positive and finite")


@dataclass(frozen=True, slots=True)
class DynamicsConstraints:
    """Kinematic limits and a simple propulsion-energy model.

    ``propagate`` returns ``None`` for an inadmissible control or a transition
    that would exceed the total energy budget. Speed saturation represents the
    vehicle controller enforcing its configured maximum speed.
    """

    max_speed_mps: float = 2.0
    max_acceleration_mps2: float = 1.0
    time_step_s: float = 1.0
    energy_budget_wh: float = 100.0
    hover_power_w: float = 80.0
    acceleration_power_w: float = 40.0

    def __post_init__(self) -> None:
        positive = (self.max_speed_mps, self.max_acceleration_mps2, self.time_step_s)
        non_negative = (self.energy_budget_wh, self.hover_power_w, self.acceleration_power_w)
        if not np.all(np.isfinite((*positive, *non_negative))):
            raise ValueError("dynamics parameters must be finite")
        if min(positive) <= 0 or min(non_negative) < 0:
            raise ValueError("kinematic limits must be positive and energy parameters non-negative")

    def propagate(self, state: PlannerState, acceleration: np.ndarray) -> PlannerState | None:
        """Apply one acceleration command while enforcing all dynamics limits."""
        control = _xyz(acceleration, "acceleration")
        control_norm = float(np.linalg.norm(control))
        if control_norm > self.max_acceleration_mps2 + _EPSILON:
            return None

        velocity = state.velocity + control * self.time_step_s
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed_mps:
            velocity *= self.max_speed_mps / speed

        position = state.position + 0.5 * (state.velocity + velocity) * self.time_step_s
        step_energy_wh = (
            (self.hover_power_w + self.acceleration_power_w * control_norm**2)
            * self.time_step_s
            / 3600.0
        )
        energy_wh = state.energy_wh + step_energy_wh
        if energy_wh > self.energy_budget_wh + _EPSILON:
            return None
        return PlannerState(position, velocity, state.time_s + self.time_step_s, energy_wh)


GridSeries = Mapping[float, np.ndarray]


def _normalize_grids(
    grids: GridSeries, name: str, *, vector: bool = False, non_negative: bool = False
) -> GridSeries:
    """Validate horizon-indexed grids and return immutable owned arrays."""
    normalized: dict[float, np.ndarray] = {}
    for raw_horizon, raw_grid in grids.items():
        horizon = float(raw_horizon)
        grid = np.asarray(raw_grid, dtype=float)
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError(f"{name} horizons must be finite and non-negative")
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"{name} grids must be finite")
        spatial_dimensions = grid.ndim - 1 if vector else grid.ndim
        if spatial_dimensions not in (1, 2, 3):
            raise ValueError(f"{name} grids must have one, two, or three spatial dimensions")
        if vector and grid.shape[-1] not in (1, 2, 3):
            raise ValueError("motion grids must end in a one-, two-, or three-component vector")
        if non_negative and np.any(grid < 0):
            raise ValueError(f"{name} grids must be non-negative")
        if not vector and name == "occupancy" and np.any((grid < 0) | (grid > 1)):
            raise ValueError("occupancy probabilities must lie in [0, 1]")
        owned = grid.copy()
        owned.flags.writeable = False
        normalized[horizon] = owned
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class PredictiveMap:
    """World-aligned probabilistic grids indexed by prediction horizon.

    Scalar grids have one to three spatial axes. Motion grids have the same
    spatial axes followed by a vector axis. Positions outside the represented
    map are assigned ``outside_risk`` instead of being silently treated as safe.
    """

    occupancy: GridSeries
    motion: GridSeries = field(default_factory=dict)
    uncertainty: GridSeries = field(default_factory=dict)
    information: GridSeries = field(default_factory=dict)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    resolution_m: float = 1.0
    outside_risk: float = 1.0

    def __post_init__(self) -> None:
        occupancy = _normalize_grids(self.occupancy, "occupancy")
        if not occupancy:
            raise ValueError("predictive map requires at least one occupancy grid")
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "motion", _normalize_grids(self.motion, "motion", vector=True))
        uncertainty = _normalize_grids(self.uncertainty, "uncertainty", non_negative=True)
        information = _normalize_grids(self.information, "information", non_negative=True)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "information", information)
        occupancy_shapes = {grid.shape for grid in occupancy.values()}
        supplemental_shapes = {grid.shape[:-1] for grid in self.motion.values()} | {
            grid.shape for grids in (uncertainty, information) for grid in grids.values()
        }
        if len(occupancy_shapes) != 1 or supplemental_shapes - occupancy_shapes:
            raise ValueError("all predictive map grids must share one spatial shape")
        object.__setattr__(self, "origin", _xyz(self.origin, "map origin"))
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("map resolution must be positive and finite")
        if not np.isfinite(self.outside_risk) or not 0 <= self.outside_risk <= 1:
            raise ValueError("outside risk must lie in [0, 1]")

    @staticmethod
    def _nearest_horizon(grids: GridSeries, time_s: float) -> float:
        """Select the nearest horizon with a deterministic earlier-time tie break."""
        if not np.isfinite(time_s):
            raise ValueError("query time must be finite")
        return min(grids, key=lambda horizon: (abs(horizon - time_s), horizon))

    def _sample(
        self,
        grids: GridSeries,
        position: np.ndarray,
        time_s: float,
        *,
        vector: bool = False,
        outside_value: float = 0.0,
    ) -> float:
        if not grids:
            return outside_value
        grid = grids[self._nearest_horizon(grids, time_s)]
        spatial_dimensions = grid.ndim - 1 if vector else grid.ndim
        coordinates = np.rint((_xyz(position, "query position") - self.origin) / self.resolution_m)
        index = tuple(coordinates[:spatial_dimensions].astype(int))
        if any(i < 0 or i >= size for i, size in zip(index, grid.shape, strict=False)):
            return outside_value
        value = grid[index]
        return float(np.linalg.norm(value)) if vector else float(value)

    def risk(self, position: np.ndarray, time_s: float) -> float:
        """Combine occupancy, motion, and uncertainty as independent hazards."""
        occupancy = self._sample(self.occupancy, position, time_s, outside_value=self.outside_risk)
        uncertainty = max(0.0, self._sample(self.uncertainty, position, time_s))
        motion = max(0.0, self._sample(self.motion, position, time_s, vector=True))
        additional_hazard = 1.0 - np.exp(-(0.35 * motion + 2.0 * uncertainty))
        return float(np.clip(1.0 - (1.0 - occupancy) * (1.0 - additional_hazard), 0.0, 1.0))

    def expected_information(self, position: np.ndarray, time_s: float) -> float:
        """Return explicit information gain, or uncertainty as a conservative proxy."""
        if self.information:
            return max(0.0, self._sample(self.information, position, time_s))
        return max(0.0, self._sample(self.uncertainty, position, time_s))


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Non-negative weights applied to the five planner cost components."""

    collision_risk: float = 12.0
    energy: float = 1.0
    time: float = 0.2
    goal_progress: float = 2.0
    information_value: float = 0.5

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in _COST_NAMES)
        if not np.all(np.isfinite(values)) or min(values) < 0:
            raise ValueError("planner cost weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Search parameters for deterministic beam planning."""

    horizon_steps: int = 8
    beam_width: int = 64
    collision_threshold: float = 0.8
    segment_samples: int = 3
    weights: CostWeights = field(default_factory=CostWeights)

    def __post_init__(self) -> None:
        for name in ("horizon_steps", "beam_width", "segment_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.collision_threshold) or not 0 < self.collision_threshold <= 1:
            raise ValueError("collision threshold must lie in (0, 1]")
        if isinstance(self.weights, Mapping):
            object.__setattr__(self, "weights", CostWeights(**self.weights))
        elif not isinstance(self.weights, CostWeights):
            raise TypeError("weights must be CostWeights or a compatible mapping")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PlannerConfig:
        """Build configuration from a JSON/YAML-style mapping."""
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Selected states, controls, aggregate costs, and terminal status."""

    trajectory: np.ndarray
    controls: np.ndarray
    cost_diagnostics: Mapping[str, float]
    reached_goal: bool

    def __post_init__(self) -> None:
        trajectory = np.asarray(self.trajectory, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3 or not np.all(np.isfinite(trajectory)):
            raise ValueError("trajectory must be a finite [steps, 3] array")
        if controls.shape != (max(len(trajectory) - 1, 0), 3) or not np.all(np.isfinite(controls)):
            raise ValueError("controls must be a finite [steps - 1, 3] array")
        object.__setattr__(self, "trajectory", trajectory.copy())
        object.__setattr__(self, "controls", controls.copy())
        diagnostics = {str(key): float(value) for key, value in self.cost_diagnostics.items()}
        if not np.all(np.isfinite(tuple(diagnostics.values()))):
            raise ValueError("cost diagnostics must be finite")
        object.__setattr__(self, "cost_diagnostics", MappingProxyType(diagnostics))


@dataclass(slots=True)
class _SearchNode:
    """Internal beam-search node."""

    cost: float
    state: PlannerState
    positions: list[np.ndarray]
    controls: list[np.ndarray]
    components: dict[str, float]


def trajectory_cost(
    previous: PlannerState,
    state: PlannerState,
    goal: PlannerGoal,
    predictive_map: PredictiveMap,
    weights: CostWeights,
) -> dict[str, float]:
    """Compute all explicit cost terms for one transition.

    Progress and expected information are benefits, so they are represented as
    negative costs. The returned ``total`` is the weighted sum of the five named
    components.
    """
    progress_m = float(
        np.linalg.norm(previous.position - goal.position)
        - np.linalg.norm(state.position - goal.position)
    )
    components = {
        "collision_risk": predictive_map.risk(state.position, state.time_s),
        "energy": state.energy_wh - previous.energy_wh,
        "time": state.time_s - previous.time_s,
        "goal_progress": -progress_m,
        "information_value": -predictive_map.expected_information(state.position, state.time_s),
    }
    components["total"] = float(
        sum(components[name] * getattr(weights, name) for name in _COST_NAMES)
    )
    return components


class TrajectoryPlanner:
    """Deterministic beam-search planner over bounded acceleration primitives."""

    def __init__(self, dynamics: DynamicsConstraints, config: PlannerConfig | None = None) -> None:
        self.dynamics = dynamics
        self.config = config or PlannerConfig()
        directions = (np.asarray(values, dtype=float) for values in product((-1, 0, 1), repeat=3))
        self._controls = tuple(
            direction / max(float(np.linalg.norm(direction)), 1.0) * dynamics.max_acceleration_mps2
            for direction in directions
        )

    def plan(
        self, initial: PlannerState, goal: PlannerGoal, predictive_map: PredictiveMap
    ) -> PlanResult:
        """Return the lowest-cost admissible trajectory within the search horizon."""
        beam = [_SearchNode(0.0, initial, [initial.position], [], dict.fromkeys(_COST_NAMES, 0.0))]
        for _ in range(self.config.horizon_steps):
            candidates = self._expand(beam, goal, predictive_map)
            if not candidates:
                break
            candidates.sort(key=lambda node: self._sort_key(node, goal))
            beam = candidates[: self.config.beam_width]
            reached = [
                node for node in beam if self._distance(node.state, goal) <= goal.tolerance_m
            ]
            if reached:
                beam = reached
                break
        return self._result(min(beam, key=lambda node: self._sort_key(node, goal)), goal)

    def _expand(
        self, beam: list[_SearchNode], goal: PlannerGoal, predictive_map: PredictiveMap
    ) -> list[_SearchNode]:
        """Generate admissible successors in a stable control order."""
        candidates: list[_SearchNode] = []
        for node in beam:
            for control in self._controls:
                next_state = self.dynamics.propagate(node.state, control)
                if next_state is None or not self._segment_is_safe(
                    node.state, next_state, predictive_map
                ):
                    continue
                parts = trajectory_cost(
                    node.state, next_state, goal, predictive_map, self.config.weights
                )
                aggregate = {name: node.components[name] + parts[name] for name in _COST_NAMES}
                candidates.append(
                    _SearchNode(
                        node.cost + parts["total"],
                        next_state,
                        [*node.positions, next_state.position],
                        [*node.controls, control],
                        aggregate,
                    )
                )
        return candidates

    def _segment_is_safe(
        self, start: PlannerState, end: PlannerState, predictive_map: PredictiveMap
    ) -> bool:
        """Sample a transition to prevent endpoint-only obstacle tunnelling."""
        for fraction in np.linspace(0.0, 1.0, self.config.segment_samples + 1)[1:]:
            position = start.position + fraction * (end.position - start.position)
            time_s = start.time_s + fraction * (end.time_s - start.time_s)
            if predictive_map.risk(position, time_s) >= self.config.collision_threshold:
                return False
        return True

    @staticmethod
    def _distance(state: PlannerState, goal: PlannerGoal) -> float:
        return float(np.linalg.norm(state.position - goal.position))

    @classmethod
    def _sort_key(cls, node: _SearchNode, goal: PlannerGoal) -> tuple[Any, ...]:
        return (node.cost, cls._distance(node.state, goal), *node.state.position.tolist())

    def _result(self, node: _SearchNode, goal: PlannerGoal) -> PlanResult:
        diagnostics = {
            **node.components,
            "total": node.cost,
            "final_distance_m": self._distance(node.state, goal),
            "energy_remaining_wh": max(0.0, self.dynamics.energy_budget_wh - node.state.energy_wh),
        }
        return PlanResult(
            np.asarray(node.positions),
            np.asarray(node.controls).reshape(-1, 3),
            diagnostics,
            diagnostics["final_distance_m"] <= goal.tolerance_m,
        )


def plan_trajectory(
    initial: PlannerState,
    goal: PlannerGoal,
    dynamics: DynamicsConstraints,
    predictive_map: PredictiveMap,
    config: PlannerConfig | None = None,
) -> PlanResult:
    """Convenience function for one deterministic planning request."""
    return TrajectoryPlanner(dynamics, config).plan(initial, goal, predictive_map)
