# Getting started

This page provides the shortest path from a clean Python environment to a validated S4D-TAM benchmark run.

## Requirements

- Python 3.10 or newer
- Git
- a virtual environment is strongly recommended

The benchmark is currently designed for research and numerical validation. The synthetic smoke configuration does not require external datasets, ROS, CUDA or flight hardware.

## Install the development environment

=== "Linux / macOS"

    ```bash
    git clone https://github.com/MatPomGit/s4d-tam.git
    cd s4d-tam
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
    ```

=== "Windows PowerShell"

    ```powershell
    git clone https://github.com/MatPomGit/s4d-tam.git
    cd s4d-tam
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
    ```

## Verify the installation

```bash
s4dtam-bench --version
s4dtam-bench doctor
```

`doctor` reports the dataset and algorithm adapter contracts available in the installed version.

Expected adapter families currently include:

- datasets: `synthetic`, `manifest`, `tartanair`, `blackbird`, `marsim`, `aeroverse`;
- algorithms: `s4dtam_reference`, `dead_reckoning`, `external_artifact`.

## Run the smoke benchmark

```bash
s4dtam-bench run configs/experiments/smoke.yaml
```

This run uses deterministic synthetic data and is intended to validate the software path, evaluator integration and report generation. It is not evidence of scientific performance.

Generated results are written under `outputs/` according to the experiment configuration. The output can include metric tables, manifests, plots, statistical summaries and failure records.

## Run an experiment configuration

The main execution interface is:

```bash
s4dtam-bench run <experiment.yaml>
```

Experiment files connect dataset definitions, algorithm configurations, seeds and reporting settings. Scientific runs should use version-controlled configuration files rather than ad-hoc command-line parameters.

## Validate an ablation configuration

```bash
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
```

Use this before running confirmatory ablations. The validation step checks the structure against the benchmark's expected ablation variants.

## Verify an independent reproduction package

```bash
s4dtam-bench verify-package <package-root> <spec.yaml>
```

This verifies an offline reproduction package against its declared specification. For independent replication, follow the full [reproduction package contract](independent-reproduction-package.md).

## Next steps

1. Read [Architecture](architecture.md) to understand the normalized contracts.
2. Read [Datasets](datasets.md) before importing real data.
3. Read [Methodology](methodology.md) and [Metrics](metrics.md) before interpreting results.
4. Use [Reproducibility](reproducibility.md) for scientific experiments.
5. Check the [Roadmap](roadmap.md) for implementation maturity and validation milestones.

!!! warning "Scientific interpretation"
    A successful smoke run means that the software path executes correctly. It does not validate the S4D-TAM research hypothesis, prove superiority over baselines or establish flight readiness.
