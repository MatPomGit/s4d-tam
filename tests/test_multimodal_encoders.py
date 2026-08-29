from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from s4dtam_benchmark.algorithms.s4dtam.encoders import (
    GNSSEncoder,
    IMUEncoder,
    LiDAREncoder,
    MaskedFusion,
    RGBEncoder,
    ThermalEncoder,
)
from s4dtam_benchmark.algorithms.s4dtam.pipeline import S4DTAMReference
from s4dtam_benchmark.contracts import AvailabilityState, RunContext, SequenceData


ENCODER_CASES = [
    (RGBEncoder, np.ones((4, 5, 3))),
    (ThermalEncoder, np.ones((4, 5))),
    (LiDAREncoder, np.ones((9, 4))),
    (IMUEncoder, np.ones(6)),
    (GNSSEncoder, np.ones(3)),
]


@pytest.mark.parametrize(("encoder_type", "sample"), ENCODER_CASES)
def test_encoder_shape_and_determinism(encoder_type, sample):
    encoder = encoder_type(output_dim=7)
    first = encoder.encode(sample, 1.25)
    second = encoder.encode(sample.copy(), 1.25)
    assert first.features.shape == (7,)
    np.testing.assert_array_equal(first.features, second.features)


def _sequence(missing: set[str] = frozenset()) -> SequenceData:
    count = 5
    streams = {
        "rgb": np.ones((count, 2, 2, 3)),
        "thermal": np.ones((count, 2, 2)),
        "lidar": np.ones((count, 3, 4)),
        "imu": np.ones((count, 6)),
        "gnss": np.ones((count, 3)),
    }
    for name in missing:
        streams[name] = None
    return SequenceData(
        dataset="test",
        sequence_id="modalities",
        timestamps=np.arange(count, dtype=float),
        gt_positions=np.zeros((count, 3)),
        **streams,
    )


@pytest.mark.parametrize("missing", [{name} for name in ("rgb", "thermal", "lidar", "imu", "gnss")])
@pytest.mark.parametrize("additional", [set(), {"rgb", "imu"}])
def test_pipeline_handles_missing_modalities(missing, additional, tmp_path: Path):
    sequence = _sequence(missing | additional)
    result = S4DTAMReference().run(sequence, RunContext(tmp_path, 1, {}))
    assert result.estimated_positions.shape == (5, 3)
    assert result.metadata["input_mode"] == "multimodal_encoded"


def test_masks_distinguish_all_unavailable_reasons(tmp_path: Path):
    sequence = _sequence({"thermal", "lidar", "imu", "gnss"})
    sequence.availability_masks["rgb"][:] = np.array(
        [
            AvailabilityState.AVAILABLE,
            AvailabilityState.SAMPLE_MISSING,
            AvailabilityState.QUALITY_REJECTED,
            AvailabilityState.AVAILABLE,
            AvailabilityState.AVAILABLE,
        ]
    )
    result = S4DTAMReference().run(sequence, RunContext(tmp_path, 1, {}))
    assert result.metadata["fused_availability_states"][:3] == [3, 1, 2]
    assert np.all(sequence.availability_masks["thermal"] == AvailabilityState.STREAM_ABSENT)


def test_fusion_excludes_rejected_observation():
    rgb = RGBEncoder(3).encode(np.ones(3), 0.0)
    thermal = ThermalEncoder(3).encode(np.full(3, 100.0), 0.0)
    fused = MaskedFusion(3).fuse(
        [rgb, thermal],
        {"rgb": AvailabilityState.AVAILABLE, "thermal": AvailabilityState.QUALITY_REJECTED},
        0.0,
    )
    np.testing.assert_array_equal(fused.features, rgb.features)


def test_sequence_rejects_inconsistent_absent_mask():
    with pytest.raises(ValueError, match="absent rgb stream"):
        SequenceData(
            "test",
            "bad",
            np.arange(2.0),
            np.zeros((2, 3)),
            availability_masks={"rgb": np.full(2, AvailabilityState.AVAILABLE)},
        )
