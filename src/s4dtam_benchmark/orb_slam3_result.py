"""Normalize frame-wise ORB-SLAM3 output into the benchmark NPZ contract."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ORB_SLAM3_V1_REVISION = "0df83dde1c85c7ab91a0d47de7a29685d046f637"
RAW_COLUMNS = (
    "timestamp",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    "latency_ms",
    "tracking_valid",
)


def normalize_orb_slam3_csv(
    raw_csv: str | Path,
    output_npz: str | Path,
    *,
    peak_rss_mb: float,
    cpu_time_s: float,
) -> Path:
    """Validate frame-wise ORB-SLAM3 CSV and emit a deterministic NPZ artifact.

    Lost frames remain on the common timestamp grid. Their pose is a finite hold-last-value
    placeholder produced by the C++ runner, while ``tracking_valid`` marks them as unavailable
    for trajectory accuracy evaluation. This prevents tracking failures from being silently
    converted into apparently valid estimates.
    """
    if not np.isfinite(peak_rss_mb) or peak_rss_mb < 0:
        raise ValueError("peak_rss_mb must be finite and non-negative")
    if not np.isfinite(cpu_time_s) or cpu_time_s < 0:
        raise ValueError("cpu_time_s must be finite and non-negative")

    source = Path(raw_csv)
    rows: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise ValueError(
                f"ORB-SLAM3 raw CSV columns must be {RAW_COLUMNS}, got {reader.fieldnames}"
            )
        rows.extend(reader)
    if not rows:
        raise ValueError("ORB-SLAM3 raw CSV is empty")

    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    positions = np.asarray(
        [[float(row[key]) for key in ("tx", "ty", "tz")] for row in rows], dtype=np.float64
    )
    quaternions = np.asarray(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows],
        dtype=np.float64,
    )
    latency = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)
    tracking_valid = np.asarray(
        [_parse_bool(row["tracking_valid"]) for row in rows], dtype=np.bool_
    )

    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("ORB-SLAM3 timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(positions)):
        raise ValueError("ORB-SLAM3 positions must be finite")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError("ORB-SLAM3 quaternions must be finite")
    if not np.all(np.isfinite(latency)) or np.any(latency < 0):
        raise ValueError("ORB-SLAM3 latency must be finite and non-negative")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms == 0):
        raise ValueError("ORB-SLAM3 quaternions must not contain zero-norm values")
    quaternions = quaternions / norms[:, None]

    destination = Path(output_npz)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        timestamps=timestamps,
        estimated_positions=positions,
        estimated_quaternions=quaternions,
        latency_ms=latency,
        tracking_valid=tracking_valid,
        alignment_mode=np.asarray("sim3"),
        resource_peak_rss_mb=np.asarray(float(peak_rss_mb)),
        resource_cpu_time_s=np.asarray(float(cpu_time_s)),
    )
    return destination


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"Invalid tracking_valid value: {value!r}")
