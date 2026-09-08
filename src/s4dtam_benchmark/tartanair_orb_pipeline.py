"""Checkpointed orchestration for the TartanAir -> ORB-SLAM3 -> S4D-TAM pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from s4dtam_benchmark.baseline_evidence import validate_and_freeze_baseline_evidence
from s4dtam_benchmark.comparison import validate_comparison_config
from s4dtam_benchmark.config import load_yaml
from s4dtam_benchmark.datasets import TartanAirDataset
from s4dtam_benchmark.experiment import run_experiment
from s4dtam_benchmark.orb_slam3_tartanair import prepare_orb_slam3_tartanair_inputs
from s4dtam_benchmark.tartanair_ingestion import convert_tartanair_v1, freeze_tartanair_cohort


PIPELINE_SCHEMA = "s4dtam-tartanair-orb-pipeline/v1"
STEPS = (
    "validate_protocol",
    "convert_tartanair",
    "preflight_tartanair",
    "freeze_tartanair",
    "prepare_orb_inputs",
    "run_orb_slam3",
    "validate_baseline_evidence",
    "run_common_benchmark",
)


@dataclass(frozen=True)
class PipelineSummary:
    state_path: Path
    completed_steps: tuple[str, ...]
    output_dir: Path | None


class PipelineBlocked(RuntimeError):
    """Raised when a scientifically required external input or result is unavailable."""


def run_tartanair_orb_pipeline(
    config_path: str | Path,
    *,
    resume: bool = True,
    dry_run: bool = False,
    until: str | None = None,
) -> PipelineSummary:
    """Execute the development vertical slice with immutable checkpoints.

    The function never fabricates baseline results. The ORB-SLAM3 command template must create
    normalized ``s4dtam-algorithm-result-npz/v1`` artifacts. Every such artifact is validated by
    the existing baseline-evidence gate before the common benchmark can run.
    """
    config_file = Path(config_path).resolve()
    config = load_yaml(config_file)
    _validate_pipeline_config(config)
    if until is not None and until not in STEPS:
        raise ValueError(f"Unknown pipeline step for --until: {until}")

    work_root = _resolve(config_file, config["work_root"])
    state_path = work_root / "pipeline-state.json"
    logs_root = work_root / "logs"
    work_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    state = _load_or_initialize_state(state_path, config_file, resume=resume)
    paths = _paths(config_file, config)

    def execute(step: str, action: Any) -> Any:
        if _step_complete(state, step) and resume:
            return None
        if dry_run:
            _record_step(state, step, "dry_run", {"note": "not executed"})
            _write_state(state_path, state)
            return None
        started = time.monotonic()
        try:
            result = action()
        except Exception as exc:
            _record_step(
                state,
                step,
                "failed",
                {
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_s": round(time.monotonic() - started, 6),
                },
            )
            _write_state(state_path, state)
            raise
        _record_step(
            state,
            step,
            "completed",
            {"elapsed_s": round(time.monotonic() - started, 6)},
        )
        _write_state(state_path, state)
        return result

    execute("validate_protocol", lambda: _validate_protocol(paths["experiment_config"]))
    if _stop_after(state, "validate_protocol", until):
        return _summary(state_path, state, None)

    execute(
        "convert_tartanair",
        lambda: convert_tartanair_v1(
            paths["raw_root"],
            paths["converted_root"],
            fps=float(config["tartanair"]["fps"]),
            link_mode=str(config["tartanair"].get("link_mode", "symlink")),
            overwrite=bool(config["tartanair"].get("overwrite_conversion", False)),
        ),
    )
    if _stop_after(state, "convert_tartanair", until):
        return _summary(state_path, state, None)

    execute("preflight_tartanair", lambda: _preflight(paths["converted_root"]))
    if _stop_after(state, "preflight_tartanair", until):
        return _summary(state_path, state, None)

    execute(
        "freeze_tartanair",
        lambda: freeze_tartanair_cohort(paths["converted_root"], paths["freeze_root"]),
    )
    if _stop_after(state, "freeze_tartanair", until):
        return _summary(state_path, state, None)

    execute(
        "prepare_orb_inputs",
        lambda: prepare_orb_slam3_tartanair_inputs(
            paths["converted_root"],
            paths["orb_input_root"],
            overwrite=bool(config["orb_slam3"].get("overwrite_inputs", False)),
        ),
    )
    if _stop_after(state, "prepare_orb_inputs", until):
        return _summary(state_path, state, None)

    execute(
        "run_orb_slam3",
        lambda: _run_orb_slam3(
            config,
            paths,
            logs_root,
            state_path=state_path,
            state=state,
        ),
    )
    if _stop_after(state, "run_orb_slam3", until):
        return _summary(state_path, state, None)

    execute(
        "validate_baseline_evidence",
        lambda: validate_and_freeze_baseline_evidence(
            baseline="orb_slam3",
            dataset="tartanair",
            sequence_list=paths["freeze_root"] / "sequence-list.txt",
            result_root=paths["orb_result_root"],
            config_path=paths["orb_config"],
            run_metadata_path=paths["run_metadata"],
            output_dir=paths["evidence_root"],
        ),
    )
    if _stop_after(state, "validate_baseline_evidence", until):
        return _summary(state_path, state, None)

    benchmark_output = execute(
        "run_common_benchmark", lambda: run_experiment(paths["experiment_config"])
    )
    output_dir = Path(benchmark_output) if benchmark_output is not None else None
    return _summary(state_path, state, output_dir)


def _validate_pipeline_config(config: dict[str, Any]) -> None:
    required = {"schema", "work_root", "tartanair", "orb_slam3", "experiment_config"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError("Pipeline configuration missing fields: " + ", ".join(missing))
    if config["schema"] != PIPELINE_SCHEMA:
        raise ValueError(f"Pipeline schema must be {PIPELINE_SCHEMA}")
    tartanair = config["tartanair"]
    if not isinstance(tartanair, dict) or not tartanair.get("raw_root"):
        raise ValueError("tartanair.raw_root is required")
    fps = float(tartanair.get("fps", 0))
    if fps <= 0:
        raise ValueError("tartanair.fps must be positive and explicitly verified")
    orb = config["orb_slam3"]
    if not isinstance(orb, dict):
        raise ValueError("orb_slam3 must be a mapping")


def _paths(config_file: Path, config: dict[str, Any]) -> dict[str, Path]:
    work_root = _resolve(config_file, config["work_root"])
    tartanair = config["tartanair"]
    orb = config["orb_slam3"]
    return {
        "raw_root": _resolve(config_file, tartanair["raw_root"]),
        "converted_root": _resolve(
            config_file, tartanair.get("converted_root", "data/raw/tartanair-converted")
        ),
        "freeze_root": _resolve(
            config_file, tartanair.get("freeze_root", "artifacts/manifests/tartanair/frozen")
        ),
        "orb_input_root": _resolve(
            config_file,
            orb.get("input_root", "artifacts/baseline-inputs/orb_slam3/tartanair"),
        ),
        "orb_result_root": _resolve(
            config_file, orb.get("result_root", "outputs/baselines/orb_slam3")
        ),
        "orb_config": _resolve(
            config_file, orb.get("config", "configs/algorithms/orb_slam3.yaml")
        ),
        "run_metadata": work_root / "orb-slam3-tartanair-run.json",
        "evidence_root": work_root / "baseline-evidence",
        "experiment_config": _resolve(config_file, config["experiment_config"]),
    }


def _resolve(config_file: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Pipeline files normally live below configs/. Resolve repository-relative paths first.
    repo_root = config_file.parent
    while repo_root.parent != repo_root and not (repo_root / "pyproject.toml").exists():
        repo_root = repo_root.parent
    if (repo_root / "pyproject.toml").exists():
        return (repo_root / path).resolve()
    return (config_file.parent / path).resolve()


def _validate_protocol(experiment_config: Path) -> None:
    protocol = load_yaml(experiment_config)
    validate_comparison_config(protocol)
    if protocol.get("study_phase") != "development":
        raise ValueError("Automated TartanAir vertical slice must use study_phase=development")
    names = {item.get("name") for item in protocol.get("algorithms", [])}
    if not {"s4d_tam_reference", "orb_slam3"}.issubset(names):
        raise ValueError("Development experiment must contain S4D-TAM and ORB-SLAM3")


def _preflight(converted_root: Path) -> None:
    sequences = list(TartanAirDataset(converted_root).sequences())
    if not sequences:
        raise ValueError("TartanAir preflight produced an empty cohort")


def _run_orb_slam3(
    config: dict[str, Any],
    paths: dict[str, Path],
    logs_root: Path,
    *,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    orb = config["orb_slam3"]
    template = str(orb.get("command_template", "")).strip()
    if not template:
        raise PipelineBlocked(
            "orb_slam3.command_template is empty. Prepare/freeze steps are complete, but a real "
            "ORB-SLAM3 execution command is required before evidence validation."
        )
    vocabulary = _resolve(Path(state["config_path"]), orb.get("vocabulary", "ORBvoc.txt"))
    if not vocabulary.is_file():
        raise PipelineBlocked(f"ORB-SLAM3 vocabulary file does not exist: {vocabulary}")

    sequence_list = [
        line.strip()
        for line in (paths["freeze_root"] / "sequence-list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result_dataset_root = paths["orb_result_root"] / "tartanair"
    result_dataset_root.mkdir(parents=True, exist_ok=True)
    commands: list[str] = []
    run_records: list[dict[str, Any]] = []

    for sequence_id in sequence_list:
        result_path = result_dataset_root / f"{sequence_id}.npz"
        if result_path.is_file() and bool(orb.get("resume_results", True)):
            run_records.append({"sequence_id": sequence_id, "status": "reused", "result": str(result_path)})
            continue
        input_dir = paths["orb_input_root"] / sequence_id
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Prepared ORB-SLAM3 input is missing: {input_dir}")
        command = template.format(
            sequence_id=shlex.quote(sequence_id),
            input_dir=shlex.quote(str(input_dir)),
            settings=shlex.quote(str(input_dir / "settings.yaml")),
            vocabulary=shlex.quote(str(vocabulary)),
            result_path=shlex.quote(str(result_path)),
        )
        commands.append(command)
        log_path = logs_root / f"orb-slam3-{sequence_id}.log"
        started = time.monotonic()
        completed = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=str(Path(state["config_path"]).parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        elapsed = time.monotonic() - started
        run_records.append(
            {
                "sequence_id": sequence_id,
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "elapsed_s": round(elapsed, 6),
                "command": command,
                "log": str(log_path),
                "result": str(result_path),
            }
        )
        state.setdefault("orb_runs", {})[sequence_id] = run_records[-1]
        _write_state(state_path, state)
        if completed.returncode != 0:
            raise RuntimeError(f"ORB-SLAM3 failed for {sequence_id}; see {log_path}")
        if not result_path.is_file():
            raise PipelineBlocked(
                f"ORB-SLAM3 command succeeded for {sequence_id} but did not create {result_path}"
            )

    freeze = json.loads((paths["freeze_root"] / "freeze.json").read_text(encoding="utf-8"))
    metadata = {
        "baseline": "orb_slam3",
        "dataset": "tartanair",
        "revision": load_yaml(paths["orb_config"])["revision"],
        "container": load_yaml(paths["orb_config"])["container"],
        "input_manifest_sha256": freeze["file_manifest_sha256"],
        "hardware": _hardware_metadata(),
        "command": template,
        "sequence_commands": commands,
        "runs": run_records,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["run_metadata"].parent.mkdir(parents=True, exist_ok=True)
    paths["run_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            metadata["nvidia_gpu"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        metadata["nvidia_gpu"] = []
    return metadata


def _load_or_initialize_state(state_path: Path, config_file: Path, *, resume: bool) -> dict[str, Any]:
    config_sha = _sha256_file(config_file)
    if state_path.is_file() and resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha:
            raise ValueError(
                "Pipeline configuration changed since the checkpoint. Use a new work_root or run "
                "with resume disabled after reviewing the change."
            )
        return state
    state = {
        "schema": PIPELINE_SCHEMA,
        "config_path": str(config_file),
        "config_sha256": config_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }
    _write_state(state_path, state)
    return state


def _record_step(state: dict[str, Any], step: str, status: str, details: dict[str, Any]) -> None:
    state.setdefault("steps", {})[step] = {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }


def _step_complete(state: dict[str, Any], step: str) -> bool:
    return state.get("steps", {}).get(step, {}).get("status") == "completed"


def _stop_after(state: dict[str, Any], step: str, until: str | None) -> bool:
    return until == step and state.get("steps", {}).get(step, {}).get("status") in {
        "completed",
        "dry_run",
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary(state_path: Path, state: dict[str, Any], output_dir: Path | None) -> PipelineSummary:
    completed = tuple(step for step in STEPS if _step_complete(state, step))
    return PipelineSummary(state_path=state_path, completed_steps=completed, output_dir=output_dir)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
