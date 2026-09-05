from __future__ import annotations

import argparse
import sys
from pathlib import Path

from s4dtam_benchmark import __version__
from s4dtam_benchmark.ablation import validate_ablation_config
from s4dtam_benchmark.baseline_evidence import validate_and_freeze_baseline_evidence
from s4dtam_benchmark.comparison import validate_comparison_config
from s4dtam_benchmark.config import load_yaml
from s4dtam_benchmark.datasets import TartanAirDataset
from s4dtam_benchmark.experiment import run_experiment
from s4dtam_benchmark.readiness import render_readiness_summary, validate_readiness_matrix
from s4dtam_benchmark.reproduction import verify_reproduction_package
from s4dtam_benchmark.study_freeze import validate_confirmatory_freeze
from s4dtam_benchmark.tartanair_ingestion import convert_tartanair_v1, freeze_tartanair_cohort


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

    convert_tartanair = subparsers.add_parser(
        "convert-tartanair",
        help="convert TartanAir V1-style pose_left/image_left trajectories to sequence.json",
    )
    convert_tartanair.add_argument("raw_root", type=Path)
    convert_tartanair.add_argument("output_root", type=Path)
    convert_tartanair.add_argument(
        "--fps",
        type=float,
        required=True,
        help="explicit sampling rate used to derive timestamps from V1 frame indices",
    )
    convert_tartanair.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="how converted frame entries materialize source RGB images",
    )
    convert_tartanair.add_argument("--overwrite", action="store_true")

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

    freeze_tartanair = subparsers.add_parser(
        "freeze-tartanair",
        help="validate and hash a converted TartanAir cohort into immutable evidence files",
    )
    freeze_tartanair.add_argument("converted_root", type=Path)
    freeze_tartanair.add_argument("output_dir", type=Path)

    baseline = subparsers.add_parser(
        "validate-baseline-evidence",
        help="validate a reproduced external baseline cohort and freeze its evidence manifest",
    )
    baseline.add_argument("baseline")
    baseline.add_argument("dataset")
    baseline.add_argument("sequence_list", type=Path)
    baseline.add_argument("result_root", type=Path)
    baseline.add_argument("config", type=Path)
    baseline.add_argument("run_metadata", type=Path)
    baseline.add_argument("output_dir", type=Path)

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
        print(
            "readiness gates: dataset-baseline matrix, TartanAir convert/preflight/freeze, "
            "baseline evidence, confirmatory freeze"
        )
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
    if args.command == "convert-tartanair":
        conversion_summary = convert_tartanair_v1(
            args.raw_root,
            args.output_root,
            fps=args.fps,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        print(
            f"Converted TartanAir: sequences={conversion_summary.sequences} "
            f"frames={conversion_summary.frames} root={conversion_summary.output_root}"
        )
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
    if args.command == "freeze-tartanair":
        freeze_summary = freeze_tartanair_cohort(args.converted_root, args.output_dir)
        print(
            f"Frozen TartanAir cohort: sequences={freeze_summary.sequences} "
            f"frames={freeze_summary.frames} files={freeze_summary.files}"
        )
        print(f"  manifest_sha256={freeze_summary.manifest_sha256}")
        print(f"  sequence_list_sha256={freeze_summary.sequence_list_sha256}")
        print(f"  output={freeze_summary.output_dir}")
        return 0
    if args.command == "validate-baseline-evidence":
        evidence_summary = validate_and_freeze_baseline_evidence(
            baseline=args.baseline,
            dataset=args.dataset,
            sequence_list=args.sequence_list,
            result_root=args.result_root,
            config_path=args.config,
            run_metadata_path=args.run_metadata,
            output_dir=args.output_dir,
        )
        print(
            f"Valid baseline evidence: baseline={evidence_summary.baseline} "
            f"dataset={evidence_summary.dataset} sequences={evidence_summary.sequences}"
        )
        print(f"  evidence={evidence_summary.evidence_path}")
        print(f"  evidence_sha256={evidence_summary.evidence_sha256}")
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
