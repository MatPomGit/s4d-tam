#!/usr/bin/env python3
"""Execute the frame-wise ORB-SLAM3 runner and normalize one sequence result."""
from __future__ import annotations

import argparse
import os
import resource
import subprocess
import tempfile
from pathlib import Path

from s4dtam_benchmark.orb_slam3_result import normalize_orb_slam3_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for label, path in (
        ("runner", args.runner),
        ("vocabulary", args.vocabulary),
        ("settings", args.settings),
        ("sequence", args.sequence),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not os.access(args.runner, os.X_OK):
        raise PermissionError(f"runner is not executable: {args.runner}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s4dtam-orb-") as temp_dir:
        raw_csv = Path(temp_dir) / "frames.csv"
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        completed = subprocess.run(
            [
                str(args.runner.resolve()),
                str(args.vocabulary.resolve()),
                str(args.settings.resolve()),
                str(args.sequence.resolve()),
                str(raw_csv),
            ],
            check=False,
            text=True,
        )
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        if completed.returncode != 0:
            return completed.returncode

        cpu_time_s = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        # Linux reports ru_maxrss in KiB. This workflow is Linux-only by design.
        peak_rss_mb = float(after.ru_maxrss) / 1024.0
        normalize_orb_slam3_csv(
            raw_csv,
            args.output,
            peak_rss_mb=peak_rss_mb,
            cpu_time_s=cpu_time_s,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
