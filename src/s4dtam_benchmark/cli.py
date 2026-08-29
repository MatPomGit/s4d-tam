from __future__ import annotations

import argparse
import sys
from pathlib import Path

from s4dtam_benchmark import __version__
from s4dtam_benchmark.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s4dtam-bench")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run an experiment YAML")
    run.add_argument("config", type=Path)
    subparsers.add_parser("doctor", help="show supported adapter contracts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print("datasets: synthetic, normalized-manifest (TartanAir/Blackbird/MARSIM/AeroVerse)")
        print("algorithms: s4dtam_reference, dead_reckoning, external_artifact")
        return 0
    output = run_experiment(args.config)
    print(f"Paper-ready artifacts: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
