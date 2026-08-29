from __future__ import annotations

import numpy as np

from s4dtam_benchmark.algorithms.s4dtam.association import FeatureAssociator, RadialAssociator
from s4dtam_benchmark.algorithms.s4dtam.memory import TokenMemory
from s4dtam_benchmark.algorithms.s4dtam.proposal import TokenProposalModule


def candidates(
    positions: list[list[float]], timestamp: float, descriptors: list[list[float]], confidence=1.0
):
    count = len(positions)
    return TokenProposalModule(2).propose(
        np.asarray(positions),
        timestamp,
        positions=np.asarray(positions),
        semantic_logits=np.tile([1.0, 0.0], (count, 1)),
        sensory_descriptors=np.asarray(descriptors),
        proposal_confidence=confidence,
    )


def test_token_identifiers_remain_stable() -> None:
    memory = TokenMemory(associator=FeatureAssociator(rejection_threshold=0.3))
    first = memory.update_candidates(candidates([[0, 0, 0]], 0.0, [[1, 0]]))[0]
    second = memory.update_candidates(candidates([[0.03, 0, 0]], 0.1, [[1, 0]]))[0]
    assert first.token_id == second.token_id == 0


def test_crossing_trajectories_use_motion_and_descriptor_evidence() -> None:
    memory = TokenMemory(associator=FeatureAssociator(rejection_threshold=0.3))
    initial = memory.update_candidates(candidates([[-1, 0, 0], [1, 0, 0]], 0, [[1, 0], [0, 1]]))
    memory.update_candidates(candidates([[-0.4, 0, 0], [0.4, 0, 0]], 1, [[1, 0], [0, 1]]))
    crossed = memory.update_candidates(candidates([[0.2, 0, 0], [-0.2, 0, 0]], 2, [[1, 0], [0, 1]]))
    assert [token.token_id for token in crossed] == [initial[0].token_id, initial[1].token_id]


def test_new_objects_and_false_low_confidence_proposals() -> None:
    memory = TokenMemory(associator=FeatureAssociator(rejection_threshold=0.6))
    memory.update_candidates(candidates([[0, 0, 0]], 0, [[1, 0]]))
    new = memory.update_candidates(candidates([[10, 0, 0]], 1, [[0, 1]]))[0]
    assert new.token_id == 1
    assert memory.update_candidates(candidates([[50, 0, 0]], 2, [[1, 1]], confidence=0.1)) == []
    assert len(memory.tokens) == 2
    assert memory.last_association.metadata["discarded_proposals"] == 1


def test_global_assignment_reports_conflicts_and_radial_fallback() -> None:
    memory = TokenMemory(associator=FeatureAssociator(rejection_threshold=0.3))
    memory.update_candidates(candidates([[0, 0, 0], [0.1, 0, 0]], 0, [[1, 0], [1, 0]]))
    memory.update_candidates(candidates([[0.04, 0, 0], [0.06, 0, 0]], 0.1, [[1, 0], [1, 0]]))
    assert memory.last_association.metadata["conflicts"]["many_to_one"] > 0
    assert memory.last_association.metadata["conflicts"]["one_to_many"] > 0

    radial = TokenMemory(associator=RadialAssociator(0.5))
    radial.update_candidates(candidates([[0, 0, 0]], 0, [[1, 0]]))
    radial.update_candidates(candidates([[0.1, 0, 0]], 1, [[1, 0]]))
    assert radial.last_association.metadata["radial_fallback_used"] is True
