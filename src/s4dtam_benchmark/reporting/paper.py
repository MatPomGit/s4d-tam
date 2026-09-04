from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from s4dtam_benchmark.evaluation.statistics import holm_adjust, paired_bootstrap


def _latex_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(str(column).replace("_", "\\_") for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells.append(rendered.replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_paper_assets(
    records: list[dict[str, Any]],
    output_dir: Path,
    run_config: dict[str, Any],
    executions: list[dict[str, Any]] | None = None,
    path_resolution: dict[str, Any] | None = None,
) -> None:
    """Write aggregate tables, figures, and an auditable run manifest.

    Args:
        records: Long-form metric records produced by sequence evaluation.
        output_dir: Destination directory for all report artifacts.
        run_config: Original, user-authored experiment configuration.
        executions: Optional per-execution metadata, including calibration provenance.
        path_resolution: Optional provided and absolute path provenance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "metrics_long.csv", index=False)
    summary = (
        frame.groupby(["dataset", "algorithm", "metric"], dropna=False)["value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "table_summary.tex").write_text(_latex_table(summary), encoding="utf-8")

    candidate = run_config.get("candidate_algorithm", "s4d_tam_reference")
    comparisons: list[dict[str, Any]] = []
    for (dataset, metric), group in frame.groupby(["dataset", "metric"]):
        pivot = group.pivot_table(index="sequence", columns="algorithm", values="value")
        if candidate not in pivot:
            continue
        for baseline in pivot.columns:
            if baseline == candidate:
                continue
            paired = pivot[[candidate, baseline]].dropna()
            if paired.empty:
                continue
            stats = paired_bootstrap(
                paired[candidate].to_numpy(),
                paired[baseline].to_numpy(),
                seed=int(run_config.get("seed", 7)),
                resamples=int(run_config.get("bootstrap_resamples", 10000)),
            )
            differences = paired[candidate].to_numpy() - paired[baseline].to_numpy()
            if len(differences) >= 2 and not np.allclose(differences, 0):
                p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
            else:
                p_value = float("nan")
            comparisons.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "candidate": candidate,
                    "baseline": baseline,
                    "n_pairs": len(paired),
                    "p_value": p_value,
                    **stats,
                }
            )
    pairwise = pd.DataFrame(comparisons)
    if not pairwise.empty:
        pairwise["p_holm"] = np.nan
        for _, indices in pairwise.groupby(["dataset", "metric"]).groups.items():
            valid = [index for index in indices if not np.isnan(pairwise.at[index, "p_value"])]
            if valid:
                adjusted = holm_adjust([float(pairwise.at[index, "p_value"]) for index in valid])
                pairwise.loc[valid, "p_holm"] = adjusted
    pairwise.to_csv(output_dir / "pairwise.csv", index=False)

    selected = frame[frame["metric"] == "trajectory/ate_rmse_m"]
    if not selected.empty:
        pivot = selected.pivot_table(
            index="algorithm", columns="dataset", values="value", aggfunc="mean"
        )
        axis = pivot.plot(kind="bar", ylabel="ATE RMSE [m]", rot=20)
        axis.grid(axis="y", alpha=0.3)
        axis.figure.tight_layout()
        axis.figure.savefig(output_dir / "ate_rmse.pdf")
        axis.figure.savefig(output_dir / "ate_rmse.png", dpi=200)
        plt.close(axis.figure)

    calibration = frame[frame["metric"].str.startswith("calibration/coverage_")]
    if not calibration.empty:
        plot = calibration.copy()
        plot["nominal"] = plot["metric"].str.extract(r"(\d+)pct")[0].astype(float) / 100
        plot = plot.groupby("nominal", as_index=False)["value"].mean().sort_values("nominal")
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
        axis.plot(plot["nominal"], plot["value"], marker="o", label="observed")
        axis.set(xlabel="Nominal coverage", ylabel="Observed coverage", xlim=(0, 1), ylim=(0, 1))
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "pose_calibration.png", dpi=200)
        figure.savefig(output_dir / "pose_calibration.pdf")
        plt.close(figure)

    execution_records = executions or []
    calibration_records = [
        {
            "algorithm": execution["algorithm"],
            **execution["calibration"],
        }
        for execution in execution_records
        if execution.get("calibration", {}).get("artifact") is not None
    ]
    # Deduplicate repeated sequence executions referencing the same model artifact.
    calibration_by_artifact = {record["artifact"]: record for record in calibration_records}
    manifest = {
        "config": run_config,
        "path_resolution": path_resolution or {},
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "executions": execution_records,
        "calibration_artifacts": list(calibration_by_artifact.values()),
        "note": "Report unavailable metrics explicitly; do not impute them.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
