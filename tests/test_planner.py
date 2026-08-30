import numpy as np

from s4dtam_benchmark.algorithms.s4dtam.planner import (
    CostWeights,
    DynamicsConstraints,
    PlannerConfig,
    PlannerGoal,
    PlannerState,
    PredictiveMap,
    plan_trajectory,
)


def _plan(grid, uncertainty=None, dynamics=None):
    return plan_trajectory(
        PlannerState(np.array([0.0, 2.0, 0.0])),
        PlannerGoal(np.array([6.0, 2.0, 0.0])),
        dynamics or DynamicsConstraints(),
        PredictiveMap({1.0: grid}, uncertainty={} if uncertainty is None else {1.0: uncertainty}),
        PlannerConfig(horizon_steps=8, beam_width=96),
    )


def test_planned_states_obey_dynamics_constraints():
    constraints = DynamicsConstraints(max_speed_mps=1.5, max_acceleration_mps2=0.75)
    result = _plan(np.zeros((9, 6)), dynamics=constraints)
    velocity = np.diff(result.trajectory, axis=0) / constraints.time_step_s
    assert np.all(np.linalg.norm(velocity, axis=1) <= constraints.max_speed_mps + 1e-12)
    if len(velocity) > 1:
        assert np.all(
            np.linalg.norm(np.diff(velocity, axis=0), axis=1)
            <= 2 * constraints.max_acceleration_mps2 + 1e-12
        )


def test_planner_avoids_occupied_cells():
    occupancy = np.zeros((9, 6))
    occupancy[2:5, 2] = 1.0
    result = _plan(occupancy)
    cells = np.floor(result.trajectory[:, :2] + 0.5).astype(int)
    assert all(
        occupancy[tuple(cell)] == 0
        for cell in cells
        if np.all((cell >= 0) & (cell < occupancy.shape))
    )


def test_uncertainty_increases_risk_and_caution():
    occupancy = np.zeros((9, 6))
    uncertainty = np.zeros_like(occupancy)
    uncertainty[1:6, 2] = 1.0
    predictive = PredictiveMap({1.0: occupancy}, uncertainty={1.0: uncertainty})
    assert predictive.risk(np.array([2.0, 2.0, 0.0]), 1.0) > predictive.risk(
        np.array([2.0, 1.0, 0.0]), 1.0
    )
    result = _plan(occupancy, uncertainty)
    assert np.any(np.abs(result.trajectory[1:-1, 1] - 2.0) > 0.1)


def test_energy_budget_is_never_exceeded():
    constraints = DynamicsConstraints(energy_budget_wh=0.03, hover_power_w=80.0)
    result = _plan(np.zeros((9, 6)), dynamics=constraints)
    consumed = constraints.energy_budget_wh - result.cost_diagnostics["energy_remaining_wh"]
    assert consumed <= constraints.energy_budget_wh + 1e-12


def test_planning_is_deterministic():
    occupancy = np.zeros((9, 6))
    first, second = _plan(occupancy), _plan(occupancy)
    np.testing.assert_array_equal(first.trajectory, second.trajectory)
    np.testing.assert_array_equal(first.controls, second.controls)
    assert first.cost_diagnostics == second.cost_diagnostics


def test_risk_uses_occupancy_motion_and_uncertainty():
    occupancy = np.full((3, 3), 0.2)
    motion = np.zeros((3, 3, 2))
    motion[1, 1] = [2.0, 0.0]
    uncertainty = np.zeros((3, 3))
    uncertainty[1, 1] = 0.3
    predictive = PredictiveMap({1.0: occupancy}, {1.0: motion}, {1.0: uncertainty})
    assert predictive.risk(np.array([1.0, 1.0, 0.0]), 1.0) > 0.2


def test_predictive_map_owns_inputs_and_treats_unknown_space_as_risky():
    occupancy = np.zeros((3, 3))
    predictive = PredictiveMap({1.0: occupancy})
    occupancy[:] = 1.0
    assert predictive.risk(np.array([1.0, 1.0, 0.0]), 1.0) == 0.0
    assert predictive.risk(np.array([10.0, 10.0, 0.0]), 1.0) == 1.0


def test_config_accepts_yaml_style_nested_cost_weights():
    config = PlannerConfig.from_mapping({"weights": {"collision_risk": 20.0}})
    assert isinstance(config.weights, CostWeights)
    assert config.weights.collision_risk == 20.0
