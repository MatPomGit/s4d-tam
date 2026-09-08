# Automated TartanAir -> ORB-SLAM3 -> S4D-TAM workflow

This document is the operational procedure for the first reproducible S4D-TAM development comparison. It complements the scientific protocol by defining exactly how data are converted, frozen, passed to ORB-SLAM3, validated and evaluated by the common benchmark.

The automated entry point is:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml
```

The pipeline is intentionally conservative. It automates deterministic preparation and validation, but it never fabricates an ORB-SLAM3 result. The baseline stage is considered complete only after the configured external command has produced valid normalized artifacts for every frozen sequence.

## 1. Scientific purpose

The workflow verifies one complete vertical slice:

```text
TartanAir upstream cohort
        |
        v
conversion to strict S4D-TAM sequence contract
        |
        v
preflight validation
        |
        v
immutable cohort freeze (SHA-256)
        |
        v
ORB-SLAM3 mono_tum input generation
        |
        v
real ORB-SLAM3 execution
        |
        v
normalized NPZ artifacts
        |
        v
baseline evidence validation
        |
        v
S4D-TAM + ORB-SLAM3 common evaluator
        |
        v
paper-ready development tables/plots
```

This is a development experiment. It must not be reported as a confirmatory H1-H7 test.

## 2. One-time environment preparation

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
```

Record the source revision used for the run:

```bash
git rev-parse HEAD
```

The exact commit should be retained with the experiment evidence.

## 3. Prepare the upstream TartanAir cohort

Place only the selected upstream trajectories below the path configured as:

```yaml
tartanair:
  raw_root: data/upstream/tartanair
```

The supported V1-style trajectory layout is:

```text
<environment>/<Easy|Hard>/<trajectory>/
  image_left/
  pose_left.txt
```

Do not modify image or pose files after the cohort is selected.

### Sampling frequency

`pose_left.txt` does not itself define per-frame timestamps. Therefore `tartanair.fps` is a scientific input, not a cosmetic option. Verify it for the exact upstream release/protocol and record the verified value in the pipeline YAML.

The supplied example uses `10.0` only as a placeholder. It must not be accepted without source verification.

## 4. Configure the pipeline

Edit:

```text
configs/pipelines/tartanair_orb_slam3_development.yaml
```

Important fields:

```yaml
schema: s4dtam-tartanair-orb-pipeline/v1
work_root: artifacts/runs/tartanair-orb-slam3-development
experiment_config: configs/experiments/tartanair_orb_slam3_development.yaml

tartanair:
  raw_root: data/upstream/tartanair
  fps: 10.0
  link_mode: symlink

orb_slam3:
  vocabulary: artifacts/baselines/orb_slam3/ORBvoc.txt
  command_template: ""
```

`work_root` is the checkpoint/evidence directory for one logical run. Use a new directory when intentionally changing the pipeline configuration or upstream cohort.

## 5. Validate the orchestration without executing external software

To inspect the command structure without doing work:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml \
  --dry-run \
  --until prepare_orb_inputs
```

For a real deterministic preparation run that stops before ORB-SLAM3:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml \
  --until prepare_orb_inputs
```

At this point the pipeline has performed the following stages.

### Step 1: `validate_protocol`

Checks that the experiment configuration is an external development comparison containing both `s4d_tam_reference` and `orb_slam3`.

### Step 2: `convert_tartanair`

Creates the strict S4D-TAM sequence descriptors, deterministic timestamps, frame materialization and stored provenance.

### Step 3: `preflight_tartanair`

Rejects malformed timestamps, missing files, invalid calibration, inconsistent frame counts, non-finite pose values and coordinate-contract violations.

### Step 4: `freeze_tartanair`

Creates:

```text
artifacts/manifests/tartanair/frozen/
  sequence-list.txt
  files.sha256
  freeze.json
```

These files define the exact cohort used by the baseline and candidate.

### Step 5: `prepare_orb_inputs`

Creates one ORB-SLAM3 input directory per frozen sequence:

```text
artifacts/baseline-inputs/orb_slam3/tartanair/<sequence-id>/
  rgb.txt
  settings.yaml
  input-manifest.json
```

Camera parameters are derived from the converted TartanAir contract. Unrelated TUM or EuRoC calibration must not be substituted.

## 6. Configure real ORB-SLAM3 execution

The baseline configuration pins the scientific identity of ORB-SLAM3:

```text
configs/algorithms/orb_slam3.yaml
```

The pipeline configuration defines how that pinned implementation is executed on the local workstation/container environment.

Required pipeline fields:

```yaml
orb_slam3:
  vocabulary: /path/to/ORBvoc.txt
  command_template: >-
    /path/to/verified/orb_wrapper
    --vocabulary {vocabulary}
    --settings {settings}
    --sequence {input_dir}
    --output {result_path}
```

Available placeholders are:

- `{sequence_id}`: frozen sequence ID;
- `{input_dir}`: prepared `mono_tum` sequence directory;
- `{settings}`: generated ORB-SLAM3 settings file;
- `{vocabulary}`: vocabulary path;
- `{result_path}`: required normalized NPZ output path.

The command is executed once for every frozen sequence.

### Required output contract

Each command must create:

```text
outputs/baselines/orb_slam3/tartanair/<sequence-id>.npz
```

The NPZ must contain at least:

```text
timestamps
estimated_positions
estimated_quaternions
latency_ms
resource_peak_rss_mb
resource_cpu_time_s
```

This requirement is deliberate. A successful ORB-SLAM3 process exit without a valid normalized artifact does not count as a reproduced baseline.

## 7. Run the complete workflow

After the vocabulary and command template are configured:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml
```

The program automatically executes the remaining stages.

### Step 6: `run_orb_slam3`

For each frozen sequence it:

1. constructs the command from the template;
2. executes the real external process;
3. captures combined stdout/stderr;
4. writes a sequence-specific log;
5. checks the return code;
6. checks that the required result file exists;
7. records elapsed execution time;
8. records sequence-level execution state in the pipeline checkpoint;
9. creates baseline run metadata including hardware information and the exact command template.

Sequence logs are stored below:

```text
<work_root>/logs/
```

If `resume_results: true`, already existing result files can be reused after an interrupted run. They are still validated later and cannot bypass the evidence gate.

### Step 7: `validate_baseline_evidence`

The existing evidence validator checks:

- exact match with the frozen sequence list;
- no missing baseline artifacts;
- no unexpected artifacts outside the frozen cohort;
- pinned ORB-SLAM3 revision;
- pinned container identity;
- valid run metadata;
- valid normalized NPZ structure;
- SHA-256 for each result artifact.

Evidence is stored under:

```text
<work_root>/baseline-evidence/
```

### Step 8: `run_common_benchmark`

Only after baseline evidence passes, the normal experiment engine runs:

```text
configs/experiments/tartanair_orb_slam3_development.yaml
```

S4D-TAM and ORB-SLAM3 are then processed by the same metric and reporting code.

## 8. Checkpoints and safe restart

The pipeline writes:

```text
<work_root>/pipeline-state.json
```

Every stage is marked as `completed`, `failed` or `dry_run` with an update timestamp and elapsed time.

Normal restart:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml
```

Completed stages are skipped. A failed stage is retried.

The state file also stores the SHA-256 of the pipeline YAML. If the configuration changes, automatic resume is rejected. This prevents a partially completed run from silently mixing two protocols.

To intentionally initialize a new state in the same work directory:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml \
  --no-resume
```

For publication work, using a new `work_root` is preferable to overwriting a prior experiment.

## 9. Run only to a selected checkpoint

The supported ordered steps are:

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

Example:

```bash
s4dtam-bench pipeline-tartanair-orb-slam3 \
  configs/pipelines/tartanair_orb_slam3_development.yaml \
  --until freeze_tartanair
```

This is useful when dataset preparation and baseline execution occur on different machines.

## 10. Reproducibility evidence to archive

For every completed development run archive at least:

```text
pipeline YAML
pipeline-state.json
repository commit SHA
TartanAir freeze.json
TartanAir sequence-list.txt
TartanAir files.sha256
ORB-SLAM3 input manifests
ORB-SLAM3 per-sequence logs
ORB-SLAM3 run metadata
ORB-SLAM3 baseline evidence JSON + SHA-256
normalized ORB-SLAM3 NPZ artifacts
common benchmark configuration
common evaluator outputs
```

Do not archive upstream datasets in the public repository unless redistribution terms explicitly permit it.

## 11. Failure rules

The workflow must stop rather than silently repair the experiment when:

- the upstream TartanAir layout is invalid;
- FPS is missing or non-positive;
- the converted cohort fails preflight;
- a frozen input is missing;
- the ORB-SLAM3 vocabulary is unavailable;
- the external command returns a non-zero status;
- the external command does not produce the expected NPZ;
- any NPZ violates the normalized result contract;
- baseline revision/container metadata do not match the pinned specification;
- the pipeline YAML changes during a resumed run.

These are scientific integrity constraints, not merely software errors.

## 12. Completion criterion

The first vertical slice is complete when `pipeline-state.json` reports all eight stages as `completed` and the baseline evidence validator has emitted a valid evidence manifest.

Even then, the result remains development evidence. Confirmatory publication evidence requires the separately frozen multi-dataset baseline matrix, H1-H7 mechanism study and field/field-analog validation defined in `manuscript/PUBLICATION_PATH.md`.
