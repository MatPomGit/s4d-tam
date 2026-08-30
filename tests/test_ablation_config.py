from copy import deepcopy
from pathlib import Path

import pytest

from s4dtam_benchmark.ablation import validate_ablation_config
from s4dtam_benchmark.config import load_yaml


CONFIG = Path(__file__).parents[1] / "configs/experiments/ablation.yaml"


def test_registered_ablation_matrix_is_complete() -> None:
    validate_ablation_config(load_yaml(CONFIG))


def test_rejects_confounded_variant() -> None:
    config = load_yaml(CONFIG)
    config["variants"][1]["overrides"]["topology"] = False
    with pytest.raises(ValueError, match="disable only semantics"):
        validate_ablation_config(config)


@pytest.mark.parametrize("trained_split", ["test", "holdout"])
def test_rejects_artifact_trained_on_evaluation_data(trained_split: str) -> None:
    config = deepcopy(load_yaml(CONFIG))
    config["evaluation_splits"] = [trained_split]
    config["artifact_provenance"]["s4dtam-full-v1"]["trained_on_splits"] = [trained_split]
    with pytest.raises(ValueError, match="evaluation/test split"):
        validate_ablation_config(config)


def test_rejects_shared_artifact() -> None:
    config = load_yaml(CONFIG)
    config["variants"][1]["artifact"] = "s4dtam-full-v1"
    with pytest.raises(ValueError, match="must not share"):
        validate_ablation_config(config)
