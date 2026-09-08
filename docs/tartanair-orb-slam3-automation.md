# Automated TartanAir -> ORB-SLAM3 -> S4D-TAM workflow

This document defines the operational procedure for the first reproducible S4D-TAM development comparison. The workflow converts and freezes TartanAir data, builds the exact ORB-SLAM3 baseline, executes it frame-by-frame, normalizes its output, validates the evidence package and finally runs the common evaluator.

The automated benchmark entry point is:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml
```

The pipeline never fabricates baseline output. ORB-SLAM3 evidence is accepted only after a real external execution has produced normalized artifacts for every frozen sequence and those artifacts pass the evidence validator.

## 1. Scientific purpose

```text
TartanAir upstream cohort
        |
        v
strict S4D-TAM conversion + preflight
        |
        v
immutable cohort freeze (SHA-256)
        |
        v
deterministic ORB-SLAM3 input generation
        |
        v
verified ORB-SLAM3 v1.0 source build
        |
        v
frame-wise monocular execution
        |
        v
normalized NPZ + tracking mask + resource measurements
        |
        v
baseline evidence validation
        |
        v
S4D-TAM + ORB-SLAM3 common evaluator
        |
        v
development tables and plots
```

This remains a development experiment. It is not confirmatory H1-H7 evidence.

## 2. Repository environment

From a clean clone:

```bash
git checkout main
git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
s4dtam-bench doctor
git rev-parse HEAD
```

Retain the repository commit SHA with every experiment package.

## 3. Build the ORB-SLAM3 baseline once

The repository uses the official ORB-SLAM3 v1.0 source revision:

```text
0df83dde1c85c7ab91a0d47de7a29685d046f637
```

The previous placeholder revision and unverified GHCR digest are not used. The execution identity is now an exact source build produced by the repository script.

Install the system dependencies required by the official ORB-SLAM3 build, including a C++ compiler, CMake, OpenCV, Eigen, Pangolin and Boost. Then run:

```bash
bash tools/orb_slam3/build_runner.sh
```

The script:

1. clones the official upstream if needed;
2. checks out the exact pinned revision in detached-HEAD mode;
3. verifies `HEAD` against the pin;
4. invokes the upstream build;
5. adds and builds the S4D-TAM frame-wise runner;
6. copies the runner and `ORBvoc.txt` into the benchmark artifact directory;
7. records SHA-256 hashes and build metadata.

Expected outputs:

```text
artifacts/baselines/orb_slam3/
  ORBvoc.txt
  build-manifest.json
  bin/
    s4dtam_orb_slam3_runner
  source/
    ... pinned upstream checkout ...
```

`build-manifest.json` is part of the scientific evidence. It binds the actual executable and vocabulary to the pinned source revision. `run_sequence.py` checks both hashes before every sequence run.

The existing configuration field named `container` is retained for compatibility with the common external-baseline evidence schema. For ORB-SLAM3 it contains the immutable source-build identity:

```text
source-build:orb-slam3-v1.0@0df83dde1c85c7ab91a0d47de7a29685d046f637
```

It must not be interpreted as a Docker image digest. If a container image is later built and published, replace this identity only with the real immutable digest produced by that build.

## 4. Prepare the TartanAir cohort

Place only the selected upstream trajectories below:

```text
data/upstream/tartanair
```

The supported V1-style layout is:

```text
<environment>/<Easy|Hard>/<trajectory>/
  image_left/
  pose_left.txt
```

Do not modify source images or poses after cohort selection.

### Sampling frequency

`pose_left.txt` does not contain per-frame timestamps. Therefore `tartanair.fps` is a scientific parameter. Verify it for the exact selected release/protocol and record the value in:

```text
configs/pipelines/tartanair_orb_slam3_development.yaml
```

The example value `10.0` is not evidence of the correct rate for every TartanAir source package.

## 5. Deterministic preparation without ORB execution

To execute all deterministic preparation stages and stop before the baseline:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml \
  --until prepare_orb_inputs
```

The stages are:

1. `validate_protocol`: verifies the development comparison contract.
2. `convert_tartanair`: generates strict sequence descriptors and timestamps.
3. `preflight_tartanair`: rejects malformed/missing inputs.
4. `freeze_tartanair`: writes immutable cohort evidence.
5. `prepare_orb_inputs`: generates ORB-SLAM3 `rgb.txt`, `settings.yaml` and an input manifest for each sequence.

Frozen cohort evidence:

```text
artifacts/manifests/tartanair/frozen/
  sequence-list.txt
  files.sha256
  freeze.json
```

Prepared ORB-SLAM3 inputs:

```text
artifacts/baseline-inputs/orb_slam3/tartanair/<sequence-id>/
  rgb.txt
  settings.yaml
  input-manifest.json
```

The camera model is derived from the TartanAir descriptor. TUM or EuRoC calibration is never substituted.

## 6. Frame-wise ORB-SLAM3 runner

`tools/orb_slam3/runner.cpp` is a non-interactive monocular runner linked against the pinned ORB-SLAM3 library. It processes every timestamp in `rgb.txt` and writes:

```text
timestamp,tx,ty,tz,qx,qy,qz,qw,latency_ms,tracking_valid
```

The runner uses `TrackMonocular` and records the tracking state after every frame. When ORB-SLAM3 loses tracking, a finite hold-last-value pose is retained only to preserve the common timestamp grid; the frame is explicitly marked `tracking_valid=0` and is excluded from trajectory-accuracy calculations. Tracking loss is therefore measurable rather than hidden.

The wrapper `tools/orb_slam3/run_sequence.py`:

1. verifies `build-manifest.json`;
2. verifies SHA-256 of the executable and vocabulary;
3. invokes the runner;
4. records process CPU time and peak RSS on Linux;
5. converts the raw CSV to the common NPZ contract.

It is normally invoked automatically by the pipeline, but a single prepared sequence can be tested manually:

```bash
python3 tools/orb_slam3/run_sequence.py \
  --runner artifacts/baselines/orb_slam3/bin/s4dtam_orb_slam3_runner \
  --vocabulary artifacts/baselines/orb_slam3/ORBvoc.txt \
  --build-manifest artifacts/baselines/orb_slam3/build-manifest.json \
  --settings artifacts/baseline-inputs/orb_slam3/tartanair/<sequence-id>/settings.yaml \
  --sequence artifacts/baseline-inputs/orb_slam3/tartanair/<sequence-id> \
  --output outputs/baselines/orb_slam3/tartanair/<sequence-id>.npz
```

## 7. Normalized result contract

Each ORB-SLAM3 result contains at least:

```text
timestamps
estimated_positions
estimated_quaternions
latency_ms
tracking_valid
alignment_mode
resource_peak_rss_mb
resource_cpu_time_s
```

`alignment_mode` is `sim3` for the monocular ORB-SLAM3 baseline. Monocular vision does not observe absolute global metric scale, so evaluating it with only rigid SE(3) alignment would penalize an unobservable gauge freedom rather than localization quality. The evaluator therefore estimates one global similarity scale for the valid trajectory before ATE/RPE translation metrics.

The estimated scale is reported as:

```text
trajectory/alignment_scale
```

Tracking robustness is reported independently through:

```text
tracking/valid_fraction
tracking/failure_rate
tracking/valid_samples
tracking/total_samples
tracking/longest_failure_run_frames
tracking/final_frame_valid
```

This keeps scale handling and tracking failure transparent rather than silently repairing the trajectory.

## 8. Run the complete pipeline

After the one-time ORB-SLAM3 build and after verifying TartanAir FPS:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml
```

The configured command template calls `run_sequence.py` once for every frozen sequence and requires a valid NPZ at:

```text
outputs/baselines/orb_slam3/tartanair/<sequence-id>.npz
```

The pipeline then performs:

- `validate_baseline_evidence`;
- `run_common_benchmark`.

A successful process exit alone is not accepted as reproduction evidence.

## 9. Baseline evidence gate

The validator checks:

- exact frozen sequence membership;
- no missing or unexpected result artifacts;
- normalized NPZ structure;
- pinned ORB-SLAM3 revision;
- immutable source-build identity;
- valid hardware/run metadata;
- per-artifact SHA-256 hashes.

Evidence is emitted below:

```text
artifacts/runs/tartanair-orb-slam3-development/baseline-evidence/
```

## 10. Checkpoints and restart

Pipeline state is stored in:

```text
artifacts/runs/tartanair-orb-slam3-development/pipeline-state.json
```

Normal restart:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml
```

Completed stages are skipped; failed stages are retried. If the pipeline YAML hash changes, automatic resume is rejected to prevent mixing two experimental protocols.

To intentionally start a new state:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml \
  --no-resume
```

For publication experiments prefer a new `work_root` instead of overwriting an old run.

## 11. Ordered pipeline stages

```text
validate_protocol
convert_tartanair
preflight_tartanair
freeze_tartanair
prepare_orb_inputs
run_orb_slam3
validate_baseline_evidence
run_common_benchmark
```

Any stage can be selected as a stopping checkpoint using `--until`.

## 12. Evidence to archive

Archive at least:

```text
repository commit SHA
pipeline YAML
pipeline-state.json
TartanAir freeze.json
TartanAir sequence-list.txt
TartanAir files.sha256
ORB-SLAM3 build-manifest.json
ORB-SLAM3 runner SHA-256
ORB-SLAM3 vocabulary SHA-256
ORB-SLAM3 input manifests
ORB-SLAM3 per-sequence logs
ORB-SLAM3 run metadata
normalized ORB-SLAM3 NPZ artifacts
baseline evidence JSON + SHA-256
common benchmark configuration
common evaluator outputs
```

Do not publish upstream dataset files unless their redistribution terms explicitly permit it.

## 13. Hard failure rules

Stop the workflow rather than silently modifying evidence when:

- TartanAir source layout is invalid;
- verified FPS is unavailable;
- preflight fails;
- a frozen file hash changes;
- the pinned ORB-SLAM3 revision cannot be checked out;
- the runner or vocabulary hash does not match the build manifest;
- the vocabulary, settings or sequence input is missing;
- ORB-SLAM3 returns a non-zero exit status;
- the expected NPZ is absent;
- timestamps are incomplete or non-monotonic;
- an NPZ violates the normalized result contract;
- baseline revision/environment identity does not match the pinned configuration;
- a resumed pipeline uses a changed YAML protocol.

These rules are scientific-integrity constraints, not only software checks.

## 14. Completion criterion

The development vertical slice is complete only when all eight pipeline stages are marked `completed`, all frozen sequences have validated ORB-SLAM3 artifacts, baseline evidence has been emitted, and the common evaluator has completed.

This still does not unlock confirmatory publication claims. The multi-dataset external comparison, H1-H7 ablations and field/field-analog validation remain separate gates defined in `manuscript/PUBLICATION_PATH.md`.
