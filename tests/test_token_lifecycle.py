from __future__ import annotations

import numpy as np
import pytest

from s4dtam_benchmark.algorithms.s4dtam.attention import (
    AttentionWeights,
    HierarchicalAttention,
)
from s4dtam_benchmark.algorithms.s4dtam.memory import (
    LifecycleRules,
    ResourceBudgets,
    TokenMemory,
)


def test_memory_growth_is_bounded_in_tokens_history_and_bytes() -> None:
    memory = TokenMemory(
        association_mode="radial",
        association_radius_m=0.01,
        budgets=ResourceBudgets(max_tokens=4, max_memory_bytes=4_000, max_history_entries=3),
    )
    for index in range(30):
        memory.update(np.array([float(index), 0.0, 0.0]), float(index))
    assert len(memory.tokens) <= 4
    assert memory.map_bytes <= 4_000
    assert all(len(token.history) <= 3 for token in memory.tokens)


def test_attention_preserves_frequently_observed_token() -> None:
    memory = TokenMemory(association_mode="radial", association_radius_m=0.2, max_tokens=2)
    important = memory.update(np.array([0.0, 0.0, 0.0]), 0.0)
    for timestamp in range(1, 6):
        memory.update(np.array([0.0, 0.0, 0.0]), float(timestamp))
    memory.update(np.array([10.0, 0.0, 0.0]), 6.0)
    memory.update(np.array([20.0, 0.0, 0.0]), 7.0)
    assert important in memory.tokens
    assert important.hit_count == 6


def test_sleeping_token_is_reactivated_during_relocalization() -> None:
    memory = TokenMemory(
        association_mode="radial",
        association_radius_m=0.5,
        lifecycle=LifecycleRules(sleep_after_s=2.0, remove_after_s=20.0),
    )
    original = memory.update(np.array([1.0, 2.0, 3.0]), 0.0)
    relocated = memory.update(np.array([1.1, 2.0, 3.0]), 3.0)
    assert relocated.token_id == original.token_id
    assert relocated.state == "active"


def test_pruning_ties_are_resolved_by_stable_identifier() -> None:
    def retained_ids() -> list[int]:
        memory = TokenMemory(association_mode="radial", association_radius_m=0.01, max_tokens=2)
        # Equal age, evidence and isolation produce an exact attention tie.
        from s4dtam_benchmark.algorithms.s4dtam.proposal import TokenProposalModule

        candidates = TokenProposalModule().propose(
            np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]), 0.0
        )
        memory.update_candidates(candidates)
        return [token.token_id for token in memory.tokens]

    assert retained_ids() == retained_ids() == [1, 2]


def test_attention_exposes_normalized_hierarchy_components() -> None:
    memory = TokenMemory(association_mode="radial")
    token = memory.update(np.array([0.0, 0.0, 0.0]), 0.0, semantic_class=2)
    scores = HierarchicalAttention().score_components([token], now_s=1.0)

    assert scores.local == {token.token_id: 0.0}
    assert 0.0 < scores.temporal[token.token_id] < 1.0
    assert 0.0 <= scores.global_[token.token_id] <= 1.0
    assert 0.0 <= scores.combined[token.token_id] <= 1.0


def test_configuration_and_timestamp_validation_are_explicit() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AttentionWeights(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="greater than"):
        LifecycleRules(sleep_after_s=2.0, remove_after_s=2.0)

    memory = TokenMemory(association_mode="radial")
    memory.update(np.zeros(3), 2.0)
    with pytest.raises(ValueError, match="monotonically"):
        memory.update(np.zeros(3), 1.0)


def test_pending_token_requires_configured_number_of_hits() -> None:
    memory = TokenMemory(
        association_mode="radial",
        lifecycle=LifecycleRules(activation_hits=3),
    )
    token = memory.update(np.zeros(3), 0.0)
    assert token.state == "pending"
    memory.update(np.zeros(3), 1.0)
    assert token.state == "pending"
    memory.update(np.zeros(3), 2.0)
    assert token.state == "active"
