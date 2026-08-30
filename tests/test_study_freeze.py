from __future__ import annotations

import copy

import pytest

from s4dtam_benchmark.study_freeze import BASELINES, DATASETS, MODELS, validate_confirmatory_freeze


def _valid_freeze() -> dict:
    sha64 = "a" * 64
    return {
        "schema_version": 1,
        "study_state": "frozen",
        "code_commit": "b" * 40,
        "preregistration": {"sha256": sha64, "external_timestamp": "registry:example"},
        "datasets": {
            name: {
                "release": f"{name}-release",
                "manifest_sha256": sha64,
                "sequence_list_sha256": sha64,
            }
            for name in DATASETS
        },
        "baselines": {
            name: {
                "validated": True,
                "config_sha256": sha64,
                "evidence_manifest_sha256": sha64,
            }
            for name in BASELINES
        },
        "models": {
            name: {"artifact_sha256": sha64, "config_sha256": sha64}
            for name in MODELS
        },
        "seeds": [2026, 2027, 2028, 2029, 2030],
        "analysis": {
            "multiplicity": "holm",
            "bootstrap_resamples": 10000,
            "h7_noninferiority_margin_pp": 2.0,
        },
    }


def test_valid_confirmatory_freeze_passes() -> None:
    validate_confirmatory_freeze(_valid_freeze())


def test_draft_freeze_is_rejected() -> None:
    config = _valid_freeze()
    config["study_state"] = "draft"
    with pytest.raises(ValueError, match="study_state"):
        validate_confirmatory_freeze(config)


def test_unvalidated_baseline_blocks_freeze() -> None:
    config = copy.deepcopy(_valid_freeze())
    config["baselines"]["orb_slam3"]["validated"] = False
    with pytest.raises(ValueError, match="orb_slam3"):
        validate_confirmatory_freeze(config)


def test_missing_h7_artifact_blocks_freeze() -> None:
    config = copy.deepcopy(_valid_freeze())
    del config["models"]["H7_no_token_lifecycle"]
    with pytest.raises(ValueError, match="H7_no_token_lifecycle"):
        validate_confirmatory_freeze(config)
