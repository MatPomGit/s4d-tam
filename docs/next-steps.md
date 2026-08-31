# Next steps

This page is the operational execution plan for the current state of S4D-TAM. It starts from a clean checkout and ends at the point where the repository can legitimately enter the frozen confirmatory study.

The order matters. Do not expose the sealed confirmatory test cohort and do not execute H1-H7 until the dataset, external-baseline and learned-model freeze gates are complete.

!!! important
    Commands below are intended for Linux or WSL with Bash and should be run from the repository root unless stated otherwise. Replace `/path/to/...` values with real paths. Never substitute dummy hashes or unpinned container tags merely to make a validator pass.

## Current executable state

The repository now supports the complete **TartanAir ingestion evidence path**:

```text
raw TartanAir V1 cohort
        |
        v
convert-tartanair
        |
        v
preflight-tartanair
        |
        v
freeze-tartanair
        |
        v
frozen sequence-list + SHA-256 manifest
        |
        v
external baseline execution
        |
        v
validate-baseline-evidence
```

The highest-priority scientific path is therefore:

```text
TartanAir -> ORB-SLAM3 -> normalized result artifact -> baseline evidence -> common evaluator
```

The repository still does **not** provide a verified one-command ORB-SLAM3 container launcher. Do not treat this as completed until the pinned external implementation has actually been executed on the frozen cohort.

## 1. Synchronize and verify the repository

Start from the latest `main`:

```bash
git checkout main
git pull --ff-only
git status --short
git rev-parse HEAD
```

Create an isolated Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the CLI and repository health:

```bash
s4dtam-bench --version
s4dtam-bench doctor
python -m compileall -q src tests
pytest
s4dtam-bench run configs/experiments/smoke.yaml
```

Validate the scientific contracts:

```bash
s4dtam-bench validate-readiness configs/readiness/dataset_baseline_matrix.yaml
s4dtam-bench validate-comparison configs/experiments/offline_benchmark.yaml
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
```

Record the exact preparation commit:

```bash
mkdir -p artifacts/freeze

git rev-parse HEAD | tee artifacts/freeze/code-commit.txt
sha256sum \
  configs/readiness/dataset_baseline_matrix.yaml \
  configs/experiments/offline_benchmark.yaml \
  configs/experiments/ablation.yaml \
  | tee artifacts/freeze/protocol-config-sha256.txt
```

**Exit criterion:** tests, smoke benchmark and all protocol validators pass from the same commit used for real-data preparation.

## 2. Prepare the research workspace

Create separate locations for upstream data, converted data, immutable evidence and baseline outputs:

```bash
mkdir -p \
  data/upstream/tartanair \
  data/upstream/blackbird \
  data/upstream/marsim \
  data/upstream/aeroverse \
  data/raw/tartanair-converted \
  data/normalized/tartanair \
  data/normalized/blackbird \
  data/normalized/marsim \
  data/normalized/aeroverse \
  artifacts/manifests/tartanair \
  artifacts/manifests/blackbird \
  artifacts/manifests/marsim \
  artifacts/manifests/aeroverse \
  artifacts/baselines \
  artifacts/models \
  outputs/baselines/orb_slam3 \
  outputs/baselines/vins_mono \
  outputs/baselines/fast_lio2 \
  outputs/baselines/lio_sam
```

Dataset files must not be committed to Git. Verify before continuing:

```bash
git status --short
```

## 3. TartanAir: acquire and freeze the first real cohort

### 3.1 Select the upstream cohort

Use the canonical TartanAir V1 layout supported by the converter:

```text
<environment>/<Easy|Hard>/<trajectory>/
  image_left/
    000000_left.png
    000001_left.png
    ...
  pose_left.txt
```

Copy only the sequences selected for this development cohort into the workspace:

```bash
export TARTANAIR_SOURCE=/path/to/downloaded/tartanair-v1

test -d "$TARTANAIR_SOURCE"
rsync -a "$TARTANAIR_SOURCE"/ data/upstream/tartanair/
```

Record an immutable upstream inventory:

```bash
find data/upstream/tartanair -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/manifests/tartanair/upstream-files.sha256

sha256sum artifacts/manifests/tartanair/upstream-files.sha256 \
  | tee artifacts/manifests/tartanair/upstream-manifest.sha256
```

### 3.2 Confirm the sampling rate

The canonical V1 `pose_left.txt` file does not encode timestamps. The converter therefore requires the frame rate to be supplied explicitly.

Determine the correct sampling rate from the exact upstream release/protocol used for the selected cohort and record it:

```bash
export TARTANAIR_FPS=10
printf '%s\n' "$TARTANAIR_FPS" \
  | tee artifacts/manifests/tartanair/declared-fps.txt
```

`10` above is only an example. Do not use it without confirming that it is correct for the selected source material.

### 3.3 Convert the raw cohort

For a local workspace, symlinks avoid duplicating RGB data:

```bash
s4dtam-bench convert-tartanair \
  data/upstream/tartanair \
  data/raw/tartanair-converted \
  --fps "$TARTANAIR_FPS" \
  --link-mode symlink
```

For a self-contained exported cohort, use copies instead:

```bash
s4dtam-bench convert-tartanair \
  data/upstream/tartanair \
  data/raw/tartanair-converted \
  --fps "$TARTANAIR_FPS" \
  --link-mode copy \
  --overwrite
```

The converter rejects:

- missing or non-contiguous frame indices;
- pose/image count mismatch;
- malformed pose rows;
- non-finite values;
- non-unit quaternions;
- invalid sampling rates.

Every generated `sequence.json` records the source trajectory, declared frame rate, materialization mode and NED-to-ENU adapter convention.

### 3.4 Run strict preflight

```bash
s4dtam-bench preflight-tartanair data/raw/tartanair-converted
```

Inspect the generated sequence IDs:

```bash
find data/raw/tartanair-converted -mindepth 2 -maxdepth 2 \
  -name sequence.json -print | sort
```

The production adapter now transforms both position and orientation ground truth consistently from TartanAir NED into ENU.

### 3.5 Freeze the converted cohort

Use the repository command instead of hand-written hashing scripts:

```bash
s4dtam-bench freeze-tartanair \
  data/raw/tartanair-converted \
  artifacts/manifests/tartanair/frozen
```

This creates:

```text
artifacts/manifests/tartanair/frozen/
  sequence-list.txt
  files.sha256
  freeze.json
```

Inspect the evidence:

```bash
cat artifacts/manifests/tartanair/frozen/freeze.json
cat artifacts/manifests/tartanair/frozen/sequence-list.txt
sha256sum artifacts/manifests/tartanair/frozen/files.sha256
```

Verify that the frozen manifest still matches the files:

```bash
(
  cd data/raw/tartanair-converted
  sha256sum -c ../../../artifacts/manifests/tartanair/frozen/files.sha256
)
```

**Exit criterion:** real selected TartanAir sequences pass conversion and preflight, `freeze.json` exists, and `files.sha256` verifies without errors.

## 4. TartanAir: reproduce ORB-SLAM3

### 4.1 Inspect the pinned baseline specification

```bash
python - <<'PY'
from s4dtam_benchmark.config import load_yaml
cfg = load_yaml("configs/algorithms/orb_slam3.yaml")
for key in ("upstream", "revision", "container", "artifact_format", "result_root"):
    print(f"{key}: {cfg.get(key)}")
PY
```

Extract the pinned image reference:

```bash
export ORB_IMAGE="$(python - <<'PY'
from s4dtam_benchmark.config import load_yaml
print(load_yaml('configs/algorithms/orb_slam3.yaml')['container'])
PY
)"

echo "$ORB_IMAGE"
```

Verify that Docker is available:

```bash
docker version
```

Then verify that the pinned image can actually be resolved:

```bash
docker pull "$ORB_IMAGE"
```

If this fails, do **not** substitute `latest`. Rebuild/publish the wrapper from the exact pinned upstream revision and update the configuration through a reviewed PR.

### 4.2 Execute only the frozen sequence list

The runner must consume exactly:

```bash
cat artifacts/manifests/tartanair/frozen/sequence-list.txt
```

Expected normalized output layout:

```text
outputs/baselines/orb_slam3/tartanair/<sequence-id>.npz
```

Each `.npz` must contain at minimum:

```text
timestamps
estimated_positions
estimated_quaternions
latency_ms
resource_peak_rss_mb
resource_cpu_time_s
```

The repository does not yet claim a verified ORB-SLAM3 launcher command because the external container entrypoint must first be reproduced and checked. This is the next external-integration implementation task after real TartanAir ingestion.

### 4.3 Record actual run metadata

After the real command has been executed, create metadata from the pinned repository configuration:

```bash
python - <<'PY'
import json
from pathlib import Path
from s4dtam_benchmark.config import load_yaml

cfg = load_yaml("configs/algorithms/orb_slam3.yaml")
freeze = json.loads(Path("artifacts/manifests/tartanair/frozen/freeze.json").read_text())
metadata = {
    "baseline": "orb_slam3",
    "dataset": "tartanair",
    "revision": cfg["revision"],
    "container": cfg["container"],
    "input_manifest_sha256": freeze["file_manifest_sha256"],
    "hardware": {
        "cpu": "REPLACE_WITH_ACTUAL_CPU",
        "gpu": "REPLACE_WITH_ACTUAL_GPU_OR_NONE",
        "ram_gb": "REPLACE_WITH_ACTUAL_RAM_GB",
    },
    "command": "REPLACE_WITH_THE_EXACT_EXECUTED_COMMAND",
}
Path("artifacts/baselines/orb-slam3-tartanair-run.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)
PY
```

Replace every `REPLACE_...` value before validation.

### 4.4 Validate and freeze ORB-SLAM3 evidence

```bash
s4dtam-bench validate-baseline-evidence \
  orb_slam3 \
  tartanair \
  artifacts/manifests/tartanair/frozen/sequence-list.txt \
  outputs/baselines/orb_slam3 \
  configs/algorithms/orb_slam3.yaml \
  artifacts/baselines/orb-slam3-tartanair-run.json \
  artifacts/baselines/evidence
```

The validator rejects:

- an unpinned revision;
- a container different from the repository contract;
- invalid input-manifest SHA-256;
- missing hardware/command metadata;
- missing frozen-sequence outputs;
- unexpected `.npz` files outside the frozen cohort;
- malformed normalized result artifacts.

Inspect the resulting evidence:

```bash
cat artifacts/baselines/evidence/orb_slam3-tartanair-evidence.json
cat artifacts/baselines/evidence/orb_slam3-tartanair-evidence.sha256
```

**Exit criterion:** the actual ORB-SLAM3 run exists for every frozen TartanAir sequence and `validate-baseline-evidence` passes.

## 5. Run the first end-to-end development comparison

Do this only after the TartanAir ORB-SLAM3 evidence is valid.

The current multi-dataset configuration expects all datasets and all configured baseline artifact roots, so do not use it blindly for the first vertical slice. Create a dedicated development experiment configuration that contains only the validated TartanAir cohort, `s4d_tam_reference` and `orb_slam3`.

Before running it, validate the external comparison contract:

```bash
s4dtam-bench validate-comparison /path/to/tartanair-orb-development.yaml
```

Then execute:

```bash
s4dtam-bench run /path/to/tartanair-orb-development.yaml
```

Inspect outputs:

```bash
find outputs -maxdepth 3 -type f -print | sort
```

This is a **development dry-run**, not a confirmatory result.

**Exit criterion:** both S4D-TAM reference and ORB-SLAM3 artifacts pass through the same evaluator and reporting path on the same frozen TartanAir sequence list.

## 6. Blackbird: visual and visual-inertial gate

Proceed only after the TartanAir vertical slice works end to end.

Copy and hash the selected Blackbird source cohort:

```bash
export BLACKBIRD_SOURCE=/path/to/downloaded/blackbird
rsync -a "$BLACKBIRD_SOURCE"/ data/upstream/blackbird/

find data/upstream/blackbird -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/manifests/blackbird/upstream-files.sha256
```

Next implementation task for this dataset:

```text
Blackbird ROS bag -> synchronized normalized SequenceData converter
```

After that converter is tested, reproduce on the same frozen Blackbird cohort:

```text
ORB-SLAM3
VINS-Mono
```

Use `validate-baseline-evidence` for each produced artifact set.

**Exit criterion:** Blackbird synchronization is tested, sequence hashes are frozen, and both applicable baseline evidence manifests pass.

## 7. MARSIM: LiDAR-inertial gate

Freeze the simulator source before generating sequences:

```bash
cd /path/to/marsim-repository
git rev-parse HEAD
git status --short
cd -
```

Record the commit:

```bash
git -C /path/to/marsim-repository rev-parse HEAD \
  | tee artifacts/manifests/marsim/source-commit.txt
```

Freeze scenario definitions and seeds before generation. Then hash generated data:

```bash
find data/upstream/marsim -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/manifests/marsim/upstream-files.sha256
```

Reproduce the compatible baselines on identical generated scenarios:

```text
FAST-LIO2
LIO-SAM
```

Validate each cohort with `validate-baseline-evidence`.

**Exit criterion:** MARSIM can be regenerated deterministically from a frozen source commit/scenario/seed set and both LiDAR-inertial baselines have validated evidence.

## 8. AeroVerse: audit before enabling

Do not change readiness states until the actual release is audited.

Create the audit record:

```bash
cat > artifacts/manifests/aeroverse/RELEASE.txt <<'EOF'
release name/version:
source URL:
retrieval date:
license:
RGB available: yes/no
IMU available: yes/no
LiDAR available: yes/no
GNSS available: yes/no
ground-truth trajectory available: yes/no
semantic labels available: yes/no
occupancy/forecast ground truth available: yes/no
EOF
```

Hash it:

```bash
sha256sum artifacts/manifests/aeroverse/RELEASE.txt \
  | tee artifacts/manifests/aeroverse/release-audit.sha256
```

Only verified sensor regimes may be changed to `blocked` or `supported` in the readiness matrix.

## 9. Revalidate readiness after every completed cohort

```bash
s4dtam-bench validate-readiness configs/readiness/dataset_baseline_matrix.yaml
pytest
```

A pair may become `supported` only when the real conversion/reproduction evidence exists.

## 10. Train and freeze learned S4D-TAM

Training and external baseline work may proceed in parallel, but the confirmatory test cohort must remain sealed.

Freeze train/calibration/test split definitions first:

```bash
mkdir -p artifacts/models/splits
sha256sum \
  /path/to/train-sequences.txt \
  /path/to/calibration-sequences.txt \
  /path/to/sealed-test-sequences.txt \
  | tee artifacts/models/splits/split-manifests.sha256
```

The publication model family must contain:

```text
full
H1_no_semantics
H2_no_temporal_state
H3_no_calibrated_uncertainty
H4_no_topology
H5_no_reference_map
H6_no_risk_prediction
H7_no_token_lifecycle
```

Before exporting the variants:

```bash
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
```

Freeze model/config artifacts:

```bash
find artifacts/models -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/models/model-files.sha256

sha256sum artifacts/models/model-files.sha256 \
  | tee artifacts/models/model-freeze.sha256
```

**Exit criterion:** `full` plus H1-H7 are immutable and differ only according to the preregistered switches.

## 11. Populate the confirmatory freeze manifest

Create the working manifest:

```bash
cp configs/reproduction/confirmatory_freeze.template.yaml \
  artifacts/freeze/confirmatory_freeze.yaml
```

Populate it only with real identifiers from completed evidence files.

Then validate:

```bash
s4dtam-bench validate-freeze artifacts/freeze/confirmatory_freeze.yaml
```

Do not proceed until this command passes without placeholders.

## 12. Execute the external confirmatory comparison

Revalidate the external protocol:

```bash
s4dtam-bench validate-comparison configs/experiments/offline_benchmark.yaml
```

Then run the sealed benchmark configuration:

```bash
s4dtam-bench run configs/experiments/offline_benchmark.yaml
```

The external study answers the complete-system competitiveness question. It must remain statistically separate from H1-H7.

## 13. Execute the internal H1-H7 study

Validate the matrix once more:

```bash
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
```

Run only the preregistered full-vs-single-component-off comparisons with the frozen seed list and test cohort.

Required analysis rules remain:

```text
paired inferential units
10,000 bootstrap resamples
Holm multiplicity correction
H7 mission-success non-inferiority margin = 2 percentage points
```

## 14. Freeze publication artifacts

Hash all final outputs used by the manuscript:

```bash
mkdir -p artifacts/publication
find outputs -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/publication/output-files.sha256

sha256sum artifacts/publication/output-files.sha256 \
  | tee artifacts/publication/output-freeze.sha256
```

Generate manuscript tables and plots only from those frozen results. Do not manually transcribe numerical results into the paper.

## 15. Progress to SIL, HIL and controlled flight

Only after the offline confirmatory study is frozen:

1. execute the SIL protocol;
2. execute the HIL protocol;
3. satisfy the safety gate for controlled real flight;
4. execute the controlled-flight protocol;
5. perform an independent clean-environment reproduction;
6. create the archival release and DOI.

Protocol pages:

```text
docs/sil-protocol.md
docs/hil-protocol.md
docs/real-flight-protocol.md
docs/independent-reproduction-package.md
```

## Immediate work queue

The next tasks, in strict order, are now:

1. obtain a pinned real TartanAir V1 cohort;
2. confirm and record its actual sampling rate;
3. run `convert-tartanair`;
4. run `preflight-tartanair`;
5. run `freeze-tartanair`;
6. verify or build the pinned ORB-SLAM3 runtime;
7. execute ORB-SLAM3 on exactly the frozen sequence list;
8. run `validate-baseline-evidence`;
9. create and execute the first TartanAir-only development comparison;
10. only after that start the Blackbird converter/reproduction gate.
