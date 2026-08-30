from __future__ import annotations

import argparse
import sys
from pathlib import Path

from s4dtam_benchmark import __version__
from s4dtam_benchmark.ablation import validate_ablation_config
from s4dtam_benchmark.comparison import validate_comparison_config
from s4dtam_benchmark.config import load_yaml
from s4dtam_benchmark.datasets import TartanAirDataset
from s4dtam_benchmark.experiment import run_experiment
from s4dtam_benchmark.readiness import render_readiness_summary, validate_readiness_matrix
from s4dtam_benchmark.reproduction import verify_reproduction_package
from s4dtam_benchmark.study_freeze import validate_confirmatory_freeze


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
    readiness = subparsers.add_parser(
        "validate-readiness",
        help="validate the dataset/sensor/baseline/metric readiness matrix",
    )
    readiness.add_argument("config", type=Path)
    freeze = subparsers.add_parser(
        "validate-freeze",
        help="validate an immutable confirmatory-study freeze manifest",
    )
    freeze.add_argument("config", type=Path)
    tartanair = subparsers.add_parser(
        "preflight-tartanair",
        help="strictly validate converted TartanAir sequence descriptors and source files",
    )
    tartanair.add_argument("root", type=Path)
    tartanair.add_argument(
        "--axis-convention",
        default="tartanair_ned_to_enu",
        choices=("tartanair_ned_to_enu", "identity"),
    )
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
        print("readiness gates: dataset-baseline matrix, TartanAir preflight, confirmatory freeze")
        return 0
    if args.command == "validate-ablation":
        validate_ablation_config(load_yaml(args.config))
        print(f"Valid ablation matrix: {args.config}")
        return 0
    if args.command == "validate-comparison":
        validate_comparison_config(load_yaml(args.config))
        print(f"Valid comparison protocol: {args.config}")
        return 0
    if args.command == "validate-readiness":
        config = load_yaml(args.config)
        validate_readiness_matrix(config)
        print(f"Valid readiness matrix: {args.config}")
        print(render_readiness_summary(config))
        return 0
    if args.command == "validate-freeze":
        validate_confirmatory_freeze(load_yaml(args.config))
        print(f"Valid confirmatory study freeze: {args.config}")
        return 0
    if args.command == "preflight-tartanair":
        sequences = list(
            TartanAirDataset(args.root, axis_convention=args.axis_convention).sequences()
        )
        frames = sum(len(sequence.timestamps) for sequence in sequences)
        print(f"Valid TartanAir conversion: sequences={len(sequences)} frames={frames}")
        for sequence in sequences:
            print(f"  {sequence.sequence_id}: samples={len(sequence.timestamps)}")
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
