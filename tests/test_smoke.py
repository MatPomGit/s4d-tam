import tempfile
import unittest
from pathlib import Path

import yaml

from s4dtam_benchmark.experiment import run_experiment


class SmokeExperimentTest(unittest.TestCase):
    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "name": "test",
                "seed": 11,
                "bootstrap_resamples": 100,
                "output_dir": str(root / "results"),
                "datasets": [{"type": "synthetic", "name": "synthetic", "length": 80}],
                "algorithms": [
                    {"type": "s4dtam_reference", "name": "s4d_tam_reference"},
                    {"type": "dead_reckoning", "name": "dead_reckoning"},
                ],
            }
            config_path = root / "experiment.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = run_experiment(config_path)
            for name in (
                "metrics_long.csv",
                "summary.csv",
                "pairwise.csv",
                "table_summary.tex",
                "run_manifest.json",
                "unavailable_metrics.json",
                "failures.json",
            ):
                self.assertTrue((output / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
