# Roadmap

The roadmap distinguishes between **software components that exist as executable research references** and **components that are complete at publication or deployment quality**. The detailed maturity boundary is described in [Project status](project-status.md) and [S4D-TAM module catalog](modules.md).

## Milestone 1: benchmark foundation

- [x] normalized dataset and algorithm contracts
- [x] executable synthetic smoke benchmark
- [x] core trajectory, semantic, forecasting, navigation, efficiency and statistical metrics
- [x] paper-oriented CSV, LaTeX, plot and run-manifest outputs
- [x] CI across supported Python versions
- [x] strict MkDocs documentation build and GitHub Pages deployment
- [x] reproduction-package and release-integrity utilities
- [x] explicit two-level comparison contract separating external benchmarking from internal H1-H7 mechanism studies

## Milestone 2: data converters and baseline reproduction

The adapter architecture exists, but public dataset and baseline validation must still be completed against exact upstream releases.

- [ ] complete TartanAir conversion with frame and calibration checks
- [ ] complete Blackbird ROS bag conversion and time-synchronization validation
- [ ] complete MARSIM scenario export with deterministic seed control
- [ ] verify AeroVerse release identity, licensing and conversion path
- [ ] reproduce pinned ORB-SLAM3 baseline configuration
- [ ] reproduce pinned VINS-Mono baseline configuration
- [ ] reproduce pinned FAST-LIO2 baseline configuration
- [ ] reproduce pinned LIO-SAM baseline configuration
- [ ] assess additional external systems only when sensing assumptions and reproduction requirements are compatible
- [ ] verify normalized external-result artifacts against common evaluators

## Milestone 3: S4D-TAM research implementation

### Executable reference components

- [x] persistent 4D token state and lifecycle infrastructure
- [x] token proposal module
- [x] feature, radial and fallback association paths
- [x] multimodal encoder interfaces and missing-modality masking infrastructure
- [x] attention-related processing module
- [x] calibration utilities and modality noise model infrastructure
- [x] reference-map representation
- [x] topological matching infrastructure
- [x] causal probabilistic occupancy and motion forecasting utilities
- [x] deterministic risk, energy, time, progress and information-aware planner
- [x] bounded token/resource budgets and lifecycle pruning
- [x] versioned token-event telemetry
- [x] integrated `S4DTAMReference` execution path

### Remaining research-quality work

- [ ] train and validate learned RGB, thermal, LiDAR, IMU and GNSS encoders
- [ ] implement and validate learned semantic token proposal where required by the final model
- [ ] implement and validate learned data association against fixed baselines
- [ ] finalize hierarchical spatiotemporal attention architecture and training procedure
- [ ] calibrate predictive covariance and out-of-distribution uncertainty on held-out data
- [ ] validate reference-map and topological matching on public sequences
- [ ] validate multi-horizon occupancy and motion forecasting
- [ ] implement a probabilistic scene-state backend supporting DBN/particle or equivalent multimodal uncertainty while preserving the causal forecast contract
- [ ] add an MPPI planning backend behind the existing planner contract and establish parity/benefit relative to deterministic beam planning
- [ ] implement adaptive octree/LOD world-model storage with risk, distance and resource-aware refinement
- [ ] implement a closed-loop active-perception controller using information gain, uncertainty, energy and compute cost
- [ ] implement safety-aware real-time scheduling with queue monitoring, deadline handling and graceful degradation
- [ ] validate risk, energy and information-aware planning objectives
- [ ] establish numerical parity tests for future optimized PyTorch/CUDA paths

## Milestone 4: confirmatory evaluation

### External system comparison

- [ ] freeze publication dataset versions and immutable sequence lists
- [ ] reproduce all selected baselines using pinned environments
- [ ] execute S4D-TAM vs ORB-SLAM3, VINS-Mono, FAST-LIO2 and LIO-SAM under the common evaluator contract
- [ ] report system-level effects separately from component-ablation inference

### Internal mechanism study

- [ ] freeze learned `full` and H1-H7 artifacts
- [ ] execute preregistered single-component ablations corresponding to hypotheses H1-H7
- [ ] generate final paired statistical comparisons and confidence intervals
- [ ] apply the preregistered Holm family correction and H7 non-inferiority rule

### Shared publication freeze

- [ ] review unavailable metrics and failure records before publication
- [ ] freeze publication figures, tables and run manifests

## Milestone 5: systems validation and release

The protocol documents exist; execution of the validation campaign remains pending.

- [x] define SIL validation protocol
- [x] define HIL validation protocol
- [x] define controlled real-flight protocol
- [x] define independent reproduction-package contract
- [ ] execute SIL campaign
- [ ] execute HIL campaign
- [ ] execute controlled real-flight campaign
- [ ] complete independent reproducibility run
- [ ] publish versioned dataset manifests and validated conversion metadata
- [ ] publish pinned container images and validated model weights
- [ ] create archival release and DOI

## Definition of research readiness

A green CI pipeline is necessary but not sufficient for publication readiness. The project should be treated as confirmatory-study ready only when dataset conversions, baseline reproductions, learned-model configuration, preregistered experiments and statistical reporting are frozen together under one versioned release.
