from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from s4dtam_benchmark.algorithms.dead_reckoning import DeadReckoning
from s4dtam_benchmark.algorithms.external import ExternalArtifactAlgorithm
from s4dtam_benchmark.algorithms.s4dtam import (
    EventLogConfig,
    LifecycleRules,
    ResourceBudgets,
    S4DTAMReference,
    ModalityNoiseModel,
)
from s4dtam_benchmark.config import load_yaml
from s4dtam_benchmark.contracts import RunContext
from s4dtam_benchmark.datasets import (
    AeroVerseDataset,
    BlackbirdDataset,
    MARSIMDataset,
    ManifestDataset,
    SyntheticDataset,
    TartanAirDataset,
)
from s4dtam_benchmark.evaluation import evaluate_result
from s4dtam_benchmark.reporting import write_paper_assets


def _dataset(spec: dict[str, Any], seed: int):
    kind = spec.get("type", spec.get("adapter", "manifest"))
    if kind == "synthetic":
        return SyntheticDataset(seed=seed, length=int(spec.get("length", 240)))
    if kind == "tartanair":
        return TartanAirDataset(spec["root"], axis_convention=spec["axis_convention"])
    if kind == "blackbird":
        return BlackbirdDataset(
            spec["root"],
            topics=spec["topics"],
            sync_tolerance_s=float(spec["sync_tolerance_s"]),
            axis_convention=spec["axis_convention"],
        )
    if kind == "marsim":
        return MARSIMDataset(spec["root"], spec.get("manifest"))
    if kind == "aeroverse":
        return AeroVerseDataset(
            spec["root"],
            required_version=spec["required_version"],
            accepted_license=spec["accepted_license"],
            manifest=spec.get("manifest"),
        )
    if kind != "manifest":
        raise ValueError(f"Unknown dataset type: {kind}")
    return ManifestDataset(spec["name"], spec["root"], spec.get("manifest"))


def _algorithm(spec: dict[str, Any]):
    kind = spec["type"]
    if kind == "s4dtam_reference":
        encoders = spec.get("encoders", {})
        lifecycle = spec.get("lifecycle", {})
        budgets = spec.get("budgets", {})
        event_logging = spec.get("event_logging", {})
        noise = spec.get("noise_model", {})
        return S4DTAMReference(
            float(spec.get("association_radius_m", 0.35)),
            int(encoders.get("output_dim", 3)),
            {
                name: float(values.get("scale", 1.0))
                for name, values in encoders.items()
                if isinstance(values, dict)
            },
            {
                name: float(weight)
                for name, weight in spec.get("fusion", {}).get("weights", {}).items()
            },
            association_mode=str(spec.get("association_mode", "feature")),
            association_rejection_threshold=float(
                spec.get("association_rejection_threshold", 0.35)
            ),
            lifecycle=LifecycleRules(
                activation_hits=int(lifecycle.get("activation_hits", 1)),
                sleep_after_s=float(lifecycle.get("sleep_after_s", 10.0)),
                reactivate_on_match=bool(lifecycle.get("reactivate_on_match", True)),
                merge_distance_m=float(lifecycle.get("merge_distance_m", 0.0)),
                remove_after_s=float(lifecycle.get("remove_after_s", 60.0)),
            ),
            budgets=ResourceBudgets(
                max_tokens=budgets.get("max_tokens"),
                max_memory_bytes=budgets.get("max_memory_bytes"),
                max_update_time_ms=budgets.get("max_update_time_ms"),
                max_history_entries=budgets.get("max_history_entries", 64),
            ),
            event_logging=EventLogConfig(
                enabled=bool(event_logging.get("enabled", True)),
                directory=str(event_logging.get("directory", "logs")),
                flush_each_event=bool(event_logging.get("flush_each_event", False)),
                include_attention_components=bool(
                    event_logging.get("include_attention_components", True)
                ),
            ),
            noise_model=ModalityNoiseModel(
                modality_variances={str(k): float(v) for k, v in
                                    noise.get("modality_variances", {}).items()},
                default_variance=float(noise.get("default_variance", 0.05)),
                process_variance_per_s=float(noise.get("process_variance_per_s", 0.01)),
                quality_power=float(noise.get("quality_power", 2.0)),
                minimum_quality=float(noise.get("minimum_quality", 0.05)),
            ),
        )
    if kind == "dead_reckoning":
        return DeadReckoning(float(spec.get("drift_per_step", 0.002)))
    if kind == "external_artifact":
        return ExternalArtifactAlgorithm(spec["name"], spec["result_root"], spec)
    raise ValueError(f"Unknown algorithm type: {kind}")


def run_experiment(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 7))
    output_dir = Path(config.get("output_dir", "outputs/run"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    context = RunContext(output_dir=output_dir, seed=seed, config=config)
    datasets = [(_dataset(spec, seed), spec) for spec in config["datasets"]]
    algorithms = [_algorithm(spec) for spec in config["algorithms"]]
    records: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    executions: list[dict[str, Any]] = []

    calibration_sequences = [sequence for dataset, spec in datasets
                             if spec.get("split") == "calibration"
                             for sequence in dataset.sequences()]
    calibration_id = ",".join(
        f"{item.dataset}/{item.sequence_id}" for item in calibration_sequences
    )
    if calibration_sequences:
        for algorithm in algorithms:
            calibrate = getattr(algorithm, "calibrate", None)
            if calibrate is not None:
                calibrate(calibration_sequences, context, calibration_id)

    for dataset, spec in datasets:
        if spec.get("split") == "calibration":
            continue
        for sequence in dataset.sequences():
            for algorithm in algorithms:
                try:
                    result = algorithm.run(sequence, context)
                    executions.append(
                        {
                            "dataset": sequence.dataset,
                            "sequence": sequence.sequence_id,
                            "algorithm": result.algorithm,
                            **result.metadata,
                        }
                    )
                    metrics, missing = evaluate_result(sequence, result)
                    records.extend(
                        {
                            "dataset": sequence.dataset,
                            "sequence": sequence.sequence_id,
                            "algorithm": result.algorithm,
                            "metric": metric,
                            "value": value,
                        }
                        for metric, value in metrics.items()
                    )
                    unavailable.extend(
                        {
                            "dataset": sequence.dataset,
                            "sequence": sequence.sequence_id,
                            "algorithm": result.algorithm,
                            "reason": reason,
                        }
                        for reason in missing
                    )
                except (
                    Exception
                ) as error:  # isolate baseline failures and keep the benchmark auditable
                    failures.append(
                        {
                            "dataset": sequence.dataset,
                            "sequence": sequence.sequence_id,
                            "algorithm": algorithm.name,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    if not records:
        raise RuntimeError(f"No successful runs. Failures: {failures}")
    write_paper_assets(records, output_dir, config, executions)
    (output_dir / "unavailable_metrics.json").write_text(
        json.dumps(unavailable, indent=2), encoding="utf-8"
    )
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return output_dir
