from __future__ import annotations

import argparse
import sys
from pathlib import Path

from s4dtam_benchmark import __version__
from s4dtam_benchmark.ablation import validate_ablation_config
from s4dtam_benchmark.comparison import validate_comparison_config
from s4dtam_benchmark.config import load_yaml
from s4dtam_benchmark.experiment import run_experiment
from s4dtam_benchmark.reproduction import verify_reproduction_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s4dtam-bench")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run an experiment YAML")
    run.add_argument("config", type=Path)
    validate = subparsers.add_parser("validate-ablation", help="validate an ablation YAML")
    validate.add_argument("config", type=Path)
    compare = subparsers.add_parser(
        "validate-comparison",
        help="validate an external benchmark or internal H1-H7 mechanism-study YAML",
    )
    compare.add_argument("config", type=Path)
    package = subparsers.add_parser("verify-package", help="verify a reproduction package")
    package.add_argument("root", type=Path)
    package.add_argument("spec", type=Path)
    subparsers.add_parser("doctor", help="show supported adapter contracts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print("datasets: synthetic, manifest, tartanair, blackbird, marsim, aeroverse")
        print("algorithms: s4dtam_reference, dead_reckoning, external_artifact")
        print("comparison levels: external, internal")
        return 0
    if args.command == "validate-ablation":
        validate_ablation_config(load_yaml(args.config))
        print(f"Valid ablation matrix: {args.config}")
        return 0
    if args.command == "validate-comparison":
        validate_comparison_config(load_yaml(args.config))
        print(f"Valid comparison protocol: {args.config}")
        return 0
    if args.command == "verify-package":
        verify_reproduction_package(args.root, load_yaml(args.spec))
        print(f"Valid reproduction package: {args.root}")
        return 0
    output = run_experiment(args.config)
    print(f"Paper-ready artifacts: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
