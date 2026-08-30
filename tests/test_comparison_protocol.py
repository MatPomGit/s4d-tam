from copy import deepcopy
from pathlib import Path

import pytest

from s4dtam_benchmark.comparison import validate_comparison_config
from s4dtam_benchmark.config import load_yaml


ROOT = Path(__file__).parents[1]
EXTERNAL = ROOT / "configs/experiments/offline_benchmark.yaml"
INTERNAL = ROOT / "configs/experiments/ablation.yaml"


def test_external_comparison_protocol_is_valid() -> None:
    validate_comparison_config(load_yaml(EXTERNAL))


def test_internal_mechanism_protocol_is_valid() -> None:
    validate_comparison_config(load_yaml(INTERNAL))


def test_external_comparison_requires_all_core_baselines() -> None:
    config = deepcopy(load_yaml(EXTERNAL))
    config["algorithms"] = [
        item for item in config["algorithms"] if item.get("name") != "lio_sam"
    ]
    with pytest.raises(ValueError, match="lio_sam"):
        validate_comparison_config(config)


def test_external_comparison_rejects_internal_ablation_matrix() -> None:
    config = deepcopy(load_yaml(EXTERNAL))
    config["variants"] = []
    with pytest.raises(ValueError, match="must not contain internal ablation"):
        validate_comparison_config(config)


def test_internal_comparison_rejects_external_algorithm_list() -> None:
    config = deepcopy(load_yaml(INTERNAL))
    config["algorithms"] = [{"name": "orb_slam3"}]
    with pytest.raises(ValueError, match="full vs H1-H7"):
        validate_comparison_config(config)
