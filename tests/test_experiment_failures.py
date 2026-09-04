import json
from pathlib import Path

import pytest
import yaml

from s4dtam_benchmark.contracts import AlgorithmResult
from s4dtam_benchmark.experiment import run_experiment


class CalibrationFailureAlgorithm:
    def __init__(self, name: str) -> None:
        self.name = name
        self.run_calls = 0

    def calibrate(self, sequences, context, data_id: str) -> None:
        raise ValueError(f"cannot calibrate {data_id}")

    def run(self, sequence, context):
        self.run_calls += 1
        raise AssertionError("an algorithm with failed calibration must not run")


class SuccessfulAlgorithm:
    name = "successful"

    def __init__(self) -> None:
        self.calibration_data_id = None
        self.run_calls = 0

    def calibrate(self, sequences, context, data_id: str) -> None:
        self.calibration_data_id = data_id

    def run(self, sequence, context) -> AlgorithmResult:
        self.run_calls += 1
        return AlgorithmResult(
            algorithm=self.name,
            timestamps=sequence.timestamps,
            estimated_positions=sequence.gt_positions,
        )


def _write_config(tmp_path: Path, algorithm_count: int) -> tuple[Path, Path]:
    output_dir = tmp_path / "results"
    config = {
        "name": "failure-isolation",
        "seed": 7,
        "bootstrap_resamples": 10,
        "output_dir": str(output_dir),
        "datasets": [
            {"type": "synthetic", "length": 12, "split": "calibration"},
            {"type": "synthetic", "length": 12, "split": "test"},
        ],
        "algorithms": [
            {"type": f"test_algorithm_{index}"} for index in range(algorithm_count)
        ],
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, output_dir


def test_calibration_failure_isolated_from_successful_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = CalibrationFailureAlgorithm("failed-calibration")
    successful = SuccessfulAlgorithm()
    algorithms = iter([failed, successful])
    monkeypatch.setattr(
        "s4dtam_benchmark.experiment._algorithm", lambda spec: next(algorithms)
    )
    config_path, output_dir = _write_config(tmp_path, algorithm_count=2)

    assert run_experiment(config_path) == output_dir

    assert failed.run_calls == 0
    assert successful.run_calls == 1
    metrics = (output_dir / "metrics_long.csv").read_text(encoding="utf-8")
    assert "successful" in metrics
    assert "failed-calibration" not in metrics
    failures = json.loads((output_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures == [
        {
            "phase": "calibration",
            "algorithm": "failed-calibration",
            "exception_type": "ValueError",
            "message": "cannot calibrate synthetic/ci_curve_001",
            "calibration_data_id": "synthetic/ci_curve_001",
        }
    ]


def test_all_calibrations_write_failures_before_final_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed_algorithms = [
        CalibrationFailureAlgorithm("failed-one"),
        CalibrationFailureAlgorithm("failed-two"),
    ]
    algorithms = iter(failed_algorithms)
    monkeypatch.setattr(
        "s4dtam_benchmark.experiment._algorithm", lambda spec: next(algorithms)
    )
    config_path, output_dir = _write_config(tmp_path, algorithm_count=2)

    with pytest.raises(RuntimeError, match="No successful runs"):
        run_experiment(config_path)

    assert all(algorithm.run_calls == 0 for algorithm in failed_algorithms)
    failure_path = output_dir / "failures.json"
    assert failure_path.is_file()
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    assert [failure["algorithm"] for failure in failures] == ["failed-one", "failed-two"]
    assert {failure["phase"] for failure in failures} == {"calibration"}
