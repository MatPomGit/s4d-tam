# S4D-TAM Benchmark

Reproducible Python benchmark and transparent reference implementation for the paper
**“S4D-TAM: Semantic 4D Token Attention Map for Autonomous Navigation of Unmanned
Aerial Vehicles in GNSS-Degraded Environments.”**

The repository compares S4D-TAM with visual, visual-inertial, and LiDAR-inertial
baselines through one normalized result contract. It produces machine-readable results,
LaTeX tables, paired-comparison tables, vector plots, run manifests, failure logs, and explicit records of metrics
that cannot be computed for a dataset.

> Research status: the included `S4DTAMReference` is an executable CPU reference of the
> token lifecycle and evaluation interfaces. It is not yet the trained hierarchical
> transformer described in the paper and is not flight-certified.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".\[dev]"
s4dtam-bench doctor
s4dtam-bench run configs/experiments/smoke.yaml
```

The smoke run is synthetic and verifies software only. Scientific experiments must use
registered dataset versions, immutable sequence lists, repeated seeds, and the protocol
in [docs/reproducibility.md](docs/reproducibility.md).

Confirmatory ablations follow the frozen [preregistration](docs/preregistration.md) and
[methodology](docs/methodology.md). Independent teams should use the validated offline
[reproduction package contract](docs/independent-reproduction-package.md), not a working
directory or an author's local environment.

## Benchmark scope

|Domain|Metrics|
|-|-|
|Localization|ATE RMSE, median and p95; translational RPE; final drift and drift percentage|
|Semantics|mIoU, class IoU, macro F1, accuracy, temporal label-flip rate|
|Forecasting|occupancy IoU, precision, recall, F1, Brier score, NLL, ECE; flow EPE|
|Navigation|mission success, collisions, collisions/km, near misses, clearance, path efficiency, energy|
|Efficiency|median/p90/p95/p99 latency, FPS, peak memory, map size, token count, energy|
|Inference|paired mission-level bootstrap CI, effect size, Holm correction utility|

Metric availability depends on ground-truth annotations. Missing values are never silently
imputed. See [docs/metrics.md](docs/metrics.md).

## Supported data sources

* TartanAir
* Blackbird UAV Dataset
* MARSIM
* AeroVerse through a manifest adapter, pending verification of the exact public release
* Any internal dataset converted to the normalized NPZ contract

Dataset files are not redistributed. See [docs/datasets.md](docs/datasets.md) for source
links, licensing checks, and conversion rules.

## Baselines

External adapters support ORB-SLAM3, VINS-Mono, FAST-LIO2, LIO-SAM, and other systems that
export the normalized result artifact. This avoids changing upstream implementations or
forcing C++/ROS systems into Python. Each baseline must run in its documented container or
environment, then export timestamps, poses, predictions, and telemetry.

## Repository map

```text
configs/                 immutable experiment, dataset, and algorithm settings
docs/                    protocol, metric definitions, data rules, roadmap
src/s4dtam\_benchmark/    reference algorithm, adapters, evaluators, reports
tests/                   numerical and end-to-end regression tests
outputs/                 ignored generated results
.github/                 CI, security analysis, issue and PR templates
```

## Citation

Use [CITATION.cff](CITATION.cff)

## License

Apache-2.0. Dataset and baseline licenses remain with their respective owners.
