from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from s4dtam_benchmark.orb_slam3_result import RAW_COLUMNS, normalize_orb_slam3_csv


def _write_raw(path: Path) -> None:
    path.write_text(
        ",".join(RAW_COLUMNS)
        + "\n"
        + "0.0,0,0,0,0,0,0,1,4.0,0\n"
        + "0.1,1,0,0,0,0,0,1,5.0,1\n"
        + "0.2,1,0,0,0,0,0,1,6.0,0\n",
        encoding="utf-8",
    )


def test_normalize_orb_slam3_csv_preserves_time_grid_and_tracking_mask(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    output = tmp_path / "result.npz"
    _write_raw(raw)

    normalize_orb_slam3_csv(raw, output, peak_rss_mb=512.0, cpu_time_s=3.5)

    with np.load(output, allow_pickle=False) as data:
        assert data["timestamps"].tolist() == [0.0, 0.1, 0.2]
        assert data["tracking_valid"].tolist() == [False, True, False]
        assert str(data["alignment_mode"]) == "sim3"
        assert float(data["resource_peak_rss_mb"]) == 512.0
        assert float(data["resource_cpu_time_s"]) == 3.5
        assert data["estimated_positions"].shape == (3, 3)
        assert data["estimated_quaternions"].shape == (3, 4)


def test_normalize_orb_slam3_csv_rejects_bad_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("timestamp,x\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        normalize_orb_slam3_csv(raw, tmp_path / "result.npz", peak_rss_mb=1, cpu_time_s=1)


def test_normalize_orb_slam3_csv_rejects_non_monotonic_time(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        ",".join(RAW_COLUMNS)
        + "\n0.1,0,0,0,0,0,0,1,1,1\n0.1,0,0,0,0,0,0,1,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_orb_slam3_csv(raw, tmp_path / "result.npz", peak_rss_mb=1, cpu_time_s=1)
