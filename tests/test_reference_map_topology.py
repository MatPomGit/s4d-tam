import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

from s4dtam_benchmark.algorithms.s4dtam import (
    CoordinateFrame,
    EventLogConfig,
    ReferenceMap,
    ReferenceToken,
    S4DTAMReference,
    TopologicalGraph,
)
from s4dtam_benchmark.contracts import AvailabilityState, RunContext, SequenceData


class ReferenceMapTest(unittest.TestCase):
    def test_coordinate_transform_and_round_trip(self):
        transform = np.eye(4)
        transform[:3, 3] = [10, -2, 1]
        map_data = ReferenceMap(
            coordinate_frames={
                "map": CoordinateFrame("map"),
                "sensor": CoordinateFrame("sensor", transform),
            },
            calibration={"camera": {"fx": 120.0}},
            origin={"latitude": 52.0, "longitude": 21.0},
            build_metadata={"builder": "unit-test"},
        )
        point = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(map_data.transform(point, "sensor"), [11, 0, 4])
        np.testing.assert_allclose(
            map_data.transform(map_data.transform(point, "sensor"), "map", "sensor"), point
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            map_data.save(path)
            self.assertEqual(ReferenceMap.load(path).to_dict(), map_data.to_dict())

    def test_loop_closure_is_verified(self):
        map_data = ReferenceMap(tokens=[ReferenceToken(4, [1, 2, 0], [1, 0, 0])])
        graph = TopologicalGraph(map_data, geometric_threshold_m=0.5)
        match, candidates, rejected = graph.match([1, 0, 0], [1.2, 2, 0])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(match.token_id, 4)
        self.assertLess(match.residual_m, 0.5)
        self.assertEqual(rejected, [])

    def test_false_and_aliased_matches_are_rejected(self):
        map_data = ReferenceMap(tokens=[
            ReferenceToken(1, [0, 0, 0], [1, 0, 0]),
            ReferenceToken(2, [0.2, 0, 0], [1, 0.01, 0]),
        ])
        graph = TopologicalGraph(map_data, ambiguity_margin=0.05, geometric_threshold_m=1.0)
        match, _, rejected = graph.match([1, 0, 0], [0.1, 0, 0])
        self.assertIsNone(match)
        self.assertTrue(all(item["reason"] == "perceptual_alias" for item in rejected))

        # Similar-looking distant places are safely disambiguated by geometry.
        map_data.tokens[1] = ReferenceToken(2, [20, 0, 0], [1, 0.01, 0])
        match, _, rejected = graph.match([1, 0, 0], [0.1, 0, 0])
        self.assertEqual(match.token_id, 1)
        self.assertEqual(rejected[0]["reason"], "geometry")

        single = TopologicalGraph(
            ReferenceMap(tokens=[ReferenceToken(1, [0, 0, 0], [1, 0, 0])]),
            geometric_threshold_m=1.0,
        )
        match, _, rejected = single.match([1, 0, 0], [5, 0, 0])
        self.assertIsNone(match)
        self.assertEqual(rejected[0]["reason"], "geometry")

    def test_pipeline_relocalizes_after_tracking_loss(self):
        reference = ReferenceMap(tokens=[ReferenceToken(7, [2, 0, 0], [1, 0, 0])])
        algorithm = S4DTAMReference(
            reference_map=reference,
            topology=TopologicalGraph(reference, geometric_threshold_m=0.5),
        )
        available = AvailabilityState.AVAILABLE
        missing = AvailabilityState.SAMPLE_MISSING
        sequence = SequenceData(
            "test", "relocalization", np.array([0.0, 1.0, 2.0]),
            np.array([[2.1, 0, 0], [2.1, 0, 0], [2.1, 0, 0]]),
            gnss=np.array([[2.1, 0, 0], [99, 0, 0], [2.1, 0, 0]]),
            availability_masks={"gnss": np.array([available, missing, available])},
            metadata={"map_descriptors": [[1, 0, 0], [0, 1, 0], [1, 0, 0]],
                      "map_positions": [[2.1, 0, 0], [2.1, 0, 0], [2.1, 0, 0]]},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = algorithm.run(sequence, RunContext(output, 1, {}))
            log_path = output / result.metadata["event_log"]
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
        correction = result.metadata["map_correction"]
        self.assertTrue(correction["relocalized"])
        self.assertEqual(correction["relocalizations"], [2])
        np.testing.assert_allclose(result.estimated_positions[2], [2, 0, 0])
        map_events = [event for event in events if event["event"].startswith("map_")]
        self.assertEqual(map_events[0]["event"], "map_mode_initialized")
        self.assertEqual(map_events[0]["map_schema"], reference.schema)
        accepted = [event for event in map_events if event["event"] == "map_match_accepted"]
        self.assertTrue(accepted[-1]["relocalized"])
        self.assertNotIn("descriptor", accepted[-1])
        completed = events[-1]
        self.assertEqual(completed["relocalization_count"], 1)

    def test_map_events_can_be_disabled_independently(self):
        reference = ReferenceMap(tokens=[ReferenceToken(1, [0, 0, 0], [1, 0, 0])])
        algorithm = S4DTAMReference(
            reference_map=reference,
            event_logging=EventLogConfig(include_map_events=False),
        )
        sequence = SequenceData(
            "test", "quiet-map", np.array([0.0]), np.zeros((1, 3)),
            observations=np.array([[0.0, 0.0, 0.0]]),
            metadata={"map_descriptors": [[1.0, 0.0, 0.0]]},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = algorithm.run(sequence, RunContext(output, 1, {}))
            events = [json.loads(line) for line in
                      (output / result.metadata["event_log"]).read_text().splitlines()]
        self.assertFalse(any(event["event"].startswith("map_") for event in events))
        self.assertNotIn("accepted_map_matches", events[-1])

    def test_mapless_mode_records_mode(self):
        algorithm = S4DTAMReference(map_enabled=False)
        sequence = SequenceData("test", "mapless", np.array([0.0]), np.zeros((1, 3)),
                                observations=np.zeros((1, 3)))
        with tempfile.TemporaryDirectory() as directory:
            result = algorithm.run(sequence, RunContext(Path(directory), 1, {}))
        self.assertEqual(result.metadata["map_correction"]["mode"], "mapless")

    def test_mapless_mode_overrides_injected_topology(self):
        reference = ReferenceMap(tokens=[ReferenceToken(1, [0, 0, 0], [1, 0, 0])])
        algorithm = S4DTAMReference(
            reference_map=reference,
            topology=TopologicalGraph(reference),
            map_enabled=False,
        )
        self.assertIsNone(algorithm.topology)


if __name__ == "__main__":
    unittest.main()
