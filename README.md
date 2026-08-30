# S4D-TAM Benchmark

[![CI](https://github.com/MatPomGit/s4d-tam/actions/workflows/ci.yml/badge.svg)](https://github.com/MatPomGit/s4d-tam/actions/workflows/ci.yml)
[![CodeQL](https://github.com/MatPomGit/s4d-tam/actions/workflows/codeql.yml/badge.svg)](https://github.com/MatPomGit/s4d-tam/actions/workflows/codeql.yml)
[![Documentation](https://github.com/MatPomGit/s4d-tam/actions/workflows/docs.yml/badge.svg)](https://matpomgit.github.io/s4d-tam/)

Reproducible benchmark and transparent Python reference implementation for **S4D-TAM: Semantic 4D Token Attention Map for Autonomous Navigation of Unmanned Aerial Vehicles in GNSS-Degraded Environments**.

The repository is designed to support publication-grade comparison of S4D-TAM with visual, visual-inertial and LiDAR-inertial navigation methods through one normalized experimental contract. It provides dataset adapters, algorithm interfaces, metrics, statistical analysis, reporting, provenance records and validation protocols.

> **Research status:** `S4DTAMReference` is an executable research reference, not the final trained hierarchical transformer and not a flight-certified navigation system. See the [project status](docs/project-status.md) and [roadmap](docs/roadmap.md).

## Documentation

The full documentation is published at **https://matpomgit.github.io/s4d-tam/**.

Start with:

- [Getting started](https://matpomgit.github.io/s4d-tam/getting-started/) for installation and the first benchmark run;
- [Project status](https://matpomgit.github.io/s4d-tam/project-status/) for implementation maturity;
- [Architecture](https://matpomgit.github.io/s4d-tam/architecture/) for the normalized benchmark contracts;
- [Methodology](https://matpomgit.github.io/s4d-tam/methodology/) and [Reproducibility](https://matpomgit.github.io/s4d-tam/reproducibility/) for scientific evaluation;
- [Datasets](https://matpomgit.github.io/s4d-tam/datasets/) and [Metrics](https://matpomgit.github.io/s4d-tam/metrics/) for experiment design.

## Quick start

```bash
git clone https://github.com/MatPomGit/s4d-tam.git
cd s4d-tam
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
s4dtam-bench doctor
s4dtam-bench run configs/experiments/smoke.yaml
```

The smoke configuration uses synthetic data. It verifies the software path, evaluators and report generation, but it is not a scientific performance result.

## What the benchmark evaluates

| Domain | Representative metrics |
| --- | --- |
| Localization | ATE RMSE, median and p95; translational RPE; final drift; drift percentage |
| Semantics | mIoU, class IoU, macro F1, accuracy, temporal label-flip rate |
| Forecasting | occupancy IoU, precision, recall, F1, Brier score, NLL, ECE, flow EPE |
| Navigation | mission success, collisions, collisions/km, near misses, clearance, path efficiency, energy |
| Efficiency | latency percentiles, FPS, peak memory, map size, token count, energy |
| Statistical inference | paired bootstrap confidence intervals, effect sizes and multiplicity correction |

Metric availability is determined by the available ground truth. Missing metrics are recorded explicitly rather than silently imputed.

## Data and baselines

The benchmark targets TartanAir, Blackbird UAV Dataset, MARSIM and AeroVerse through dataset-specific or manifest-based adapters. Internal datasets can be evaluated after conversion to the normalized contract. Dataset files are not redistributed by this repository.

External baseline integration is based on a normalized result artifact. This allows systems such as ORB-SLAM3, VINS-Mono, FAST-LIO2 and LIO-SAM to run in their native C++/ROS environments while being evaluated by the same Python metrics and reporting stack.

## Current S4D-TAM reference modules

The source tree already contains executable components for token lifecycle management, proposal and association, multimodal encoder interfaces, attention-related processing, calibration, reference-map and topological support, forecasting, planning and token-event telemetry. These components form a transparent research reference and evaluation target; they should not be interpreted as proof that the final learned architecture is complete or validated.

## Reproducible experiment workflow

```text
Dataset / manifest
       ↓
Normalized SequenceData
       ↓
S4D-TAM or external baseline
       ↓
Normalized AlgorithmResult
       ↓
Common evaluators
       ↓
Statistics + provenance + failure records
       ↓
CSV / LaTeX / plots / publication artifacts
```

Scientific experiments should use version-controlled configurations, immutable sequence lists, repeated seeds and the protocol described in [docs/reproducibility.md](docs/reproducibility.md). Confirmatory ablations should follow [docs/preregistration.md](docs/preregistration.md).

## Repository structure

```text
configs/                  experiment, dataset and algorithm configuration
docs/                     MkDocs documentation and validation protocols
src/s4dtam_benchmark/     benchmark core, algorithms, adapters and reporting
tests/                    numerical, contract and regression tests
tools/                    release and research-support utilities
outputs/                  generated benchmark results, ignored by Git
.github/workflows/        CI, documentation deployment and security analysis
```

## Development and CI

The active-development CI verifies installation, syntax and unit/regression tests on Python 3.10 and 3.12. The smoke benchmark runs on `main`. Ruff findings are currently advisory so that accumulated style debt does not mask numerical failures. Documentation is built with MkDocs in strict mode and deployed through GitHub Pages.

See [CONTRIBUTING.md](CONTRIBUTING.md) before modifying scientific contracts, metrics, dataset registration or experiment protocols.

## Citation

Citation metadata is maintained in [CITATION.cff](CITATION.cff). The documentation also provides a dedicated [citation page](https://matpomgit.github.io/s4d-tam/citation/).

## License

The project is distributed under the Apache-2.0 license. Dataset and baseline licenses remain with their respective owners.
