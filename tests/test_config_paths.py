import json
import csv
import io
from pathlib import Path

import numpy as np
import yaml

from s4dtam_benchmark.config import load_yaml, resolve_from_config
from s4dtam_benchmark.experiment import run_experiment


def test_resolve_from_config_accepts_path_and_optional_value(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "experiment.yaml"
    config_path.parent.mkdir()
    config_path.write_text("name: paths\n", encoding="utf-8")
    config = load_yaml(config_path)

    assert resolve_from_config(config, Path("../data")) == (tmp_path / "data").resolve()
    assert resolve_from_config(config, None) is None
    assert resolve_from_config(config, tmp_path / "absolute") == (tmp_path / "absolute").resolve()


def test_relative_experiment_paths_do_not_depend_on_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "portable-experiment"
    data_root = project / "inputs"
    data_root.mkdir(parents=True)
    np.savez(
        data_root / "sequence.npz",
        timestamps=np.array([0.0, 1.0, 2.0]),
        gt_positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    (project / "dataset.json").write_text(
        json.dumps({"sequences": [{"id": "same-input", "file": "sequence.npz"}]}),
        encoding="utf-8",
    )
    config = {
        "name": "cwd-independent",
        "seed": 3,
        "bootstrap_resamples": 10,
        "output_dir": "artifacts/run",
        "datasets": [
            {
                "type": "manifest",
                "name": "fixture",
                "root": "inputs",
                "manifest": "dataset.json",
            }
        ],
        "algorithms": [{"type": "dead_reckoning", "name": "dead_reckoning"}],
    }
    config_path = project / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifests = []
    metrics = []
    for cwd in (Path.cwd(), tmp_path):
        monkeypatch.chdir(cwd)
        output = run_experiment(config_path)
        assert output == (project / "artifacts/run").resolve()
        manifests.append(json.loads((output / "run_manifest.json").read_text()))
        rows = csv.DictReader(io.StringIO((output / "metrics_long.csv").read_text()))
        metrics.append(
            [row for row in rows if not row["metric"].startswith("efficiency/")]
        )

    assert metrics[0] == metrics[1]
    assert manifests[0]["config"] == config
    assert manifests[0]["path_resolution"] == manifests[1]["path_resolution"]
    paths = manifests[0]["path_resolution"]
    assert paths["datasets"][0]["root"] == {
        "provided": "inputs",
        "resolved": str(data_root.resolve()),
    }
    assert paths["datasets"][0]["manifest"] == {
        "provided": "dataset.json",
        "resolved": str((project / "dataset.json").resolve()),
    }
    assert paths["output_dir"]["provided"] == "artifacts/run"
    assert paths["output_dir"]["resolved"] == str((project / "artifacts/run").resolve())
