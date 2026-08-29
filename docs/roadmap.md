# Roadmap

## Milestone 1: benchmark foundation

- [x] normalized dataset and algorithm contracts
- [x] executable synthetic smoke benchmark
- [x] core trajectory, semantic, forecast, navigation, efficiency, and statistical metrics
- [x] paper-oriented CSV, LaTeX, PDF, PNG, and run-manifest outputs
- [x] CI and numerical unit tests

## Milestone 2: data converters and baseline wrappers

- [ ] TartanAir converter with frame and calibration checks
- [ ] Blackbird ROS bag converter and time synchronization validation
- [ ] MARSIM scenario exporter and deterministic seed control
- [ ] AeroVerse adapter after release and license verification
- [ ] pinned wrappers for ORB-SLAM3, VINS-Mono, FAST-LIO2, and LIO-SAM

## Milestone 3: full S4D-TAM

- [ ] RGB, thermal, LiDAR, IMU, and GNSS encoders with missing-modality masks
- [ ] semantic token proposal and learned data association
- [ ] hierarchical spatiotemporal attention and token pruning
- [ ] calibrated covariance and out-of-distribution uncertainty
- [ ] reference-map and topological matching
- [ ] multi-horizon occupancy and motion forecasting
- [ ] risk, energy, and information-aware planner

## Milestone 4: validation and release

- [ ] preregistered ablations corresponding to hypotheses H1-H7
- [ ] SIL, HIL, and real-flight safety protocol
- [ ] independent reproducibility run
- [ ] versioned dataset manifests, container images, model weights, and DOI release
