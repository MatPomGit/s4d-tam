# Next steps

This page is the operational execution plan for the current state of S4D-TAM. It starts from a clean checkout and ends at the point where the repository can legitimately enter the frozen confirmatory study. The order matters: do not expose the sealed test cohort or run H1-H7 before the data, baseline and model freeze gates are complete.

!!! important
    Commands below are intended for Linux or WSL with Bash. Run them from the repository root unless a step explicitly says otherwise. Replace values written as `/path/to/...` with real local paths. Do not replace blocked evidence fields with dummy hashes just to make a validator pass.

## Execution order

1. synchronize and verify the repository;
2. validate the existing scientific contracts;
3. prepare and freeze the real TartanAir cohort;
4. reproduce ORB-SLAM3 on exactly that cohort;
5. prepare Blackbird and reproduce ORB-SLAM3 and VINS-Mono;
6. prepare MARSIM and reproduce FAST-LIO2 and LIO-SAM;
7. audit AeroVerse before enabling any external pair;
8. train and calibrate the learned S4D-TAM model using train/calibration data only;
9. freeze `full` and H1-H7 model artifacts;
10. populate and validate the confirmatory freeze manifest;
11. only then execute the sealed external comparison and internal H1-H7 study;
12. generate publication artifacts and proceed to SIL, HIL and controlled flight validation.

The current highest-priority path is therefore **TartanAir -> ORB-SLAM3 -> normalized result artifact -> common evaluator**.

## 1. Synchronize the repository

Start from the latest `main` and create an isolated research environment.

```bash
git checkout main
git pull --ff-only

git status --short
git rev-parse HEAD

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installed command and supported contracts:

```bash
s4dtam-bench --version
s4dtam-bench doctor
```

Run the local software gate before touching real data:

```bash
python -m compileall -q src tests
pytest
s4dtam-bench run configs/experiments/smoke.yaml
```

**Exit criterion:** tests and the synthetic smoke benchmark pass from the same commit that will be used to prepare real-data artifacts.

## 2. Validate the scientific contracts

Validate the dataset/baseline readiness matrix and the preregistered H1-H7 matrix.

```bash
s4dtam-bench validate-readiness configs/readiness/dataset_baseline_matrix.yaml
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
```

Record the exact code commit used for preparation:

```bash
mkdir -p artifacts/freeze

git rev-parse HEAD | tee artifacts/freeze/code-commit.txt
sha256sum configs/readiness/dataset_baseline_matrix.yaml \
  configs/experiments/ablation.yaml \
  configs/experiments/offline_benchmark.yaml \
  | tee artifacts/freeze/protocol-config-sha256.txt
```

Do **not** run the final multi-dataset benchmark yet. `configs/experiments/offline_benchmark.yaml` expects normalized public datasets and external baseline artifacts that are not yet reproduced.

**Exit criterion:** readiness and ablation validators pass and the preparation commit/configuration hashes are recorded.

## 3. Prepare the data workspace

Keep downloaded upstream files, converted files, normalized benchmark files and immutable manifests separate.

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
  artifacts/manifests \
  artifacts/baselines \
  artifacts/models \
  outputs/baselines/orb_slam3 \
  outputs/baselines/vins_mono \
  outputs/baselines/fast_lio2 \
  outputs/baselines/lio_sam
```

The repository intentionally does not redistribute dataset files. Acquire each dataset from its official upstream source and preserve the exact release/archive identifiers used. Do not download a moving `latest` snapshot for a confirmatory run.

## 4. TartanAir: freeze the first real cohort

### 4.1 Copy the selected upstream data into the workspace

After downloading the selected TartanAir sequences using the official upstream mechanism, point `TARTANAIR_SOURCE` at the downloaded directory and copy it without modifying the source files:

```bash
export TARTANAIR_SOURCE=/path/to/downloaded/tartanair

test -d "$TARTANAIR_SOURCE"
rsync -a "$TARTANAIR_SOURCE"/ data/upstream/tartanair/
```

Record a deterministic inventory and SHA-256 hashes:

```bash
find data/upstream/tartanair -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/manifests/tartanair-upstream-files.sha256

sha256sum artifacts/manifests/tartanair-upstream-files.sha256 \
  | tee artifacts/manifests/tartanair-upstream-manifest.sha256
```

### 4.2 Convert the selected sequences to the strict TartanAir adapter contract

The strict adapter requires one `sequence.json` per sequence. Each descriptor must contain ordered frames, timestamps in seconds, positions in metres and complete calibration metadata. The contract is implemented in `src/s4dtam_benchmark/datasets/tartanair.py`.

At the current repository state there is **no official raw-TartanAir-to-`sequence.json` CLI converter yet**. Implementing that converter is the next small software task before real TartanAir ingestion can be automated. Until it exists, do not hand-edit a large cohort and call it reproducible.

After the converter has produced `data/raw/tartanair-converted`, verify that descriptors exist:

```bash
find data/raw/tartanair-converted -name sequence.json -print | sort
```

Then run the strict preflight:

```bash
s4dtam-bench preflight-tartanair data/raw/tartanair-converted
```

### 4.3 Freeze the sequence list and converted cohort

```bash
find data/raw/tartanair-converted -name sequence.json -print \
  | sort \
  > artifacts/manifests/tartanair-sequences.txt

sha256sum artifacts/manifests/tartanair-sequences.txt \
  | tee artifacts/manifests/tartanair-sequences.sha256

find data/raw/tartanair-converted -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/manifests/tartanair-converted-files.sha256

sha256sum artifacts/manifests/tartanair-converted-files.sha256 \
  | tee artifacts/manifests/tartanair-converted-manifest.sha256
```

**Exit criterion:** the real converted cohort passes `preflight-tartanair`, the exact sequence list is frozen, and both upstream and converted-file manifests have SHA-256 identifiers.

## 5. TartanAir: reproduce ORB-SLAM3

TartanAir is currently the first visual baseline gate. Inspect the pinned ORB-SLAM3 specification before execution:

```bash
python - <<'PY'
from s4dtam_benchmark.config import load_yaml
cfg = load_yaml("configs/algorithms/orb_slam3.yaml")
for key in ("upstream", "revision", "container", "artifact_format", "result_root"):
    print(f"{key}: {cfg.get(key)}")
PY
```

Verify that the configured container reference is resolvable before relying on it:

```bash
export ORB_IMAGE="$(python - <<'PY'
from s4dtam_benchmark.config import load_yaml
print(load_yaml('configs/algorithms/orb_slam3.yaml')['container'])
PY
)"

echo "$ORB_IMAGE"
docker pull "$ORB_IMAGE"
```

If the pull fails, **do not change the configured digest to an unpinned tag**. Build and publish a reproducible image from the pinned upstream revision, then update the configuration in a reviewed PR.

The repository currently defines ORB-SLAM3 as an `external_artifact`; it does not yet provide a verified one-command launcher that feeds the frozen TartanAir cohort into ORB-SLAM3. The next implementation deliverable is therefore a baseline runner that:

1. reads the frozen sequence list;
2. applies the exact camera/calibration mapping from `configs/algorithms/orb_slam3.yaml`;
3. executes the pinned implementation/container;
4. writes one normalized `s4dtam-algorithm-result-npz/v1` artifact per sequence;
5. records hardware, runtime, container digest, configuration hash and input manifest hash.

After the runner exists, verify that outputs are present only for the frozen sequence set:

```bash
find outputs/baselines/orb_slam3 -type f -print | sort
```

Freeze the reproduced artifacts:

```bash
find outputs/baselines/orb_slam3 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/baselines/orb-slam3-files.sha256

sha256sum artifacts/baselines/orb-slam3-files.sha256 \
  configs/algorithms/orb_slam3.yaml \
  | tee artifacts/baselines/orb-slam3-evidence.sha256
```

**Exit criterion:** ORB-SLAM3 has actually run on the frozen TartanAir cohort, normalized artifacts pass the common result contract, and an evidence manifest records the exact inputs, revision/container, configuration and hardware.

## 6. Blackbird: visual and visual-inertial gate

The Blackbird stage begins only after the TartanAir -> ORB-SLAM3 vertical slice is working end-to-end.

Copy the pinned Blackbird source files and freeze their hashes:

```bash
export BLACKBIRD_SOURCE=/path/to/downloaded/blackbird

test -d "$BLACKBIRD_SOURCE"
rsync -a "$BLACKBIRD_SOURCE"/ data/upstream/blackbird/

find data/upstream/blackbird -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/manifests/blackbird-upstream-files.sha256

sha256sum artifacts/manifests/blackbird-upstream-files.sha256 \
  | tee artifacts/manifests/blackbird-upstream-manifest.sha256
```

Then complete and test the Blackbird ROS bag/time-synchronization converter before producing `data/normalized/blackbird`.

Inspect the two applicable baseline specifications:

```bash
python - <<'PY'
from s4dtam_benchmark.config import load_yaml
for name in ("orb_slam3", "vins_mono"):
    cfg = load_yaml(f"configs/algorithms/{name}.yaml")
    print(name, cfg["revision"], cfg["container"], sep="\n  ")
PY
```

Reproduce ORB-SLAM3 and VINS-Mono on the **same frozen Blackbird sequence list**, then store normalized artifacts under:

```text
outputs/baselines/orb_slam3/
outputs/baselines/vins_mono/
```

**Exit criterion:** Blackbird synchronization is verified and both compatible baselines have normalized, hashed evidence artifacts.

## 7. MARSIM: LiDAR-inertial gate

Freeze the exact MARSIM commit, scenario definitions and random seeds before generating trajectories. The simulation must be reproducible from those identifiers.

Create an evidence directory:

```bash
mkdir -p artifacts/manifests/marsim
```

Record the chosen source commit from the checked-out MARSIM repository:

```bash
cd /path/to/marsim-repository
git rev-parse HEAD | tee /tmp/marsim-commit.txt
git status --short
cd -

cp /tmp/marsim-commit.txt artifacts/manifests/marsim/source-commit.txt
```

Hash exported source data/scenarios after deterministic generation:

```bash
find data/upstream/marsim -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/manifests/marsim-upstream-files.sha256

sha256sum artifacts/manifests/marsim-upstream-files.sha256 \
  artifacts/manifests/marsim/source-commit.txt \
  | tee artifacts/manifests/marsim-release-evidence.sha256
```

Inspect the applicable LiDAR-inertial baselines:

```bash
python - <<'PY'
from s4dtam_benchmark.config import load_yaml
for name in ("fast_lio2", "lio_sam"):
    cfg = load_yaml(f"configs/algorithms/{name}.yaml")
    print(name, cfg["revision"], cfg["container"], sep="\n  ")
PY
```

Then reproduce FAST-LIO2 and LIO-SAM using identical frozen MARSIM scenarios/seeds and normalize their outputs.

**Exit criterion:** deterministic MARSIM regeneration has been demonstrated and both LiDAR-inertial baseline artifact sets are frozen and validated.

## 8. AeroVerse: audit before use

Do not enable AeroVerse pairs merely because an adapter exists. First verify the exact public release identity, license, downloadable assets and actual sensor modalities.

Create an audit record:

```bash
mkdir -p artifacts/manifests/aeroverse

touch artifacts/manifests/aeroverse/RELEASE.txt
```

Record in `RELEASE.txt` at minimum:

```text
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
```

Hash the completed audit and downloaded files:

```bash
sha256sum artifacts/manifests/aeroverse/RELEASE.txt \
  | tee artifacts/manifests/aeroverse/release-audit.sha256
```

Only after this audit should `configs/readiness/dataset_baseline_matrix.yaml` be changed from blocked/not-applicable for newly verified sensor regimes.

**Exit criterion:** release identity and licensing are defensible and every enabled dataset/baseline pair is supported by actual released sensor data.

## 9. Revalidate dataset/baseline readiness

After each real-data or baseline milestone, rerun:

```bash
s4dtam-bench validate-readiness configs/readiness/dataset_baseline_matrix.yaml
pytest
```

A pair should move to `supported` only after its conversion/reproduction evidence exists. A pinned configuration by itself is not sufficient.

## 10. Train the learned S4D-TAM model

This work may proceed in parallel with external baseline reproduction, but training must use only the predefined train split and hyperparameter/covariance calibration only the calibration split. The confirmatory test cohort remains sealed.

Before training, record the split manifests:

```bash
mkdir -p artifacts/models/splits

sha256sum /path/to/train-sequences.txt \
  /path/to/calibration-sequences.txt \
  /path/to/sealed-test-sequences.txt \
  | tee artifacts/models/splits/split-manifests.sha256
```

The final training pipeline must produce immutable artifacts for:

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

Validate the H1-H7 experiment definition again before exporting those variants:

```bash
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
```

Freeze every model/config artifact:

```bash
find artifacts/models -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/models/model-files.sha256

sha256sum artifacts/models/model-files.sha256 \
  | tee artifacts/models/model-freeze.sha256
```

**Exit criterion:** `full` and all seven single-component-off variants are immutable, share the frozen training protocol and differ only in the preregistered component switch.

## 11. Populate the confirmatory freeze manifest

Create the real manifest from the intentionally invalid template:

```bash
cp configs/reproduction/confirmatory_freeze.template.yaml \
  artifacts/freeze/confirmatory_freeze.yaml
```

Fill it only with real identifiers collected in previous steps. Required evidence includes code commit, preregistration SHA/timestamp, all four dataset manifests and sequence lists, baseline evidence, `full` plus H1-H7 artifacts, seeds and statistical settings.

Validate it:

```bash
s4dtam-bench validate-freeze artifacts/freeze/confirmatory_freeze.yaml
```

If this command fails, the confirmatory study is not ready. Fix the missing evidence rather than weakening the validator.

Freeze the accepted manifest itself:

```bash
sha256sum artifacts/freeze/confirmatory_freeze.yaml \
  | tee artifacts/freeze/confirmatory_freeze.sha256
```

**Exit criterion:** `validate-freeze` succeeds without placeholders.

## 12. Execute the sealed external benchmark

Only after the freeze succeeds should the sealed test cohort become available to the execution environment.

First confirm that all expected normalized datasets and baseline output roots exist:

```bash
for path in \
  data/normalized/tartanair \
  data/normalized/blackbird \
  data/normalized/marsim \
  data/normalized/aeroverse \
  outputs/baselines/orb_slam3 \
  outputs/baselines/vins_mono \
  outputs/baselines/fast_lio2 \
  outputs/baselines/lio_sam; do
  test -e "$path" || { echo "Missing: $path"; exit 1; }
done
```

Then execute the external comparison:

```bash
s4dtam-bench run configs/experiments/offline_benchmark.yaml
```

Immediately freeze generated results:

```bash
find outputs/offline_multidataset -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/freeze/offline-results.sha256
```

**Exit criterion:** the external system comparison completes using only sensor-compatible pairs and all result/manifests are immutable.

## 13. Execute H1-H7 as a separate internal study

Do not merge the external baseline comparison with component-ablation inference. H1-H7 answers causal mechanism questions inside S4D-TAM and must use the frozen `full` model plus one-component-off variants.

Before execution:

```bash
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
s4dtam-bench validate-freeze artifacts/freeze/confirmatory_freeze.yaml
```

Run only through the frozen internal-study configuration produced for the learned artifacts. Preserve one result row per statistical unit/mission rather than treating frames as independent observations.

Required final analysis settings remain:

```text
family alpha: 0.05
multiplicity correction: Holm
bootstrap: BCa, 10000 resamples
H7 mission-success non-inferiority margin: 2 percentage points
```

**Exit criterion:** H1-H7 outputs are complete, failure records are preserved and the preregistered statistical rules have not been changed after test disclosure.

## 14. Publication freeze

Before copying numbers into the manuscript, archive the exact result set used for tables and figures.

```bash
mkdir -p artifacts/publication

find outputs -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/publication/output-files.sha256

sha256sum \
  artifacts/freeze/confirmatory_freeze.yaml \
  artifacts/publication/output-files.sha256 \
  | tee artifacts/publication/publication-freeze.sha256
```

Then review:

- unavailable metrics and why they are unavailable;
- failed sequences and failure handling;
- external-system results separately from H1-H7 inference;
- confidence intervals and corrected inferential decisions;
- hardware/runtime provenance for latency and memory claims;
- exact manuscript tables/figures against frozen CSV outputs.

## 15. SIL, HIL and controlled flight

Only after the offline model and protocol are frozen should the systems-validation campaign progress through the existing protocol documents:

1. [SIL protocol](sil-protocol.md);
2. [HIL protocol](hil-protocol.md);
3. [controlled real-flight protocol](real-flight-protocol.md).

The same immutable model/configuration identity should be carried forward unless a safety-critical correction is required. Any such change creates a new version and invalidates direct comparison with the previous frozen campaign.

## What should be implemented next in the repository

The immediate coding backlog is intentionally narrow:

1. **TartanAir conversion CLI** that creates the strict `sequence.json` descriptors from pinned upstream sequences and records provenance;
2. **external baseline runner contract** for ORB-SLAM3 first, including frozen sequence input, calibration mapping, container/revision verification, normalized result export and evidence manifest;
3. **normalized external artifact validator** to reject incomplete or mismatched baseline outputs before the common evaluator runs;
4. Blackbird converter/synchronization validation;
5. MARSIM deterministic export utility;
6. analogous reproducible runners for VINS-Mono, FAST-LIO2 and LIO-SAM;
7. only then expand the learned-model training backend needed for the final `full` and H1-H7 artifacts.

This ordering gives the project a complete real-data vertical slice as early as possible instead of implementing every dataset and every model component in parallel without an end-to-end reproducibility proof.

## Minimal checklist for the next work session

If only one work session is available, execute these steps in order:

```bash
git checkout main
git pull --ff-only
source .venv/bin/activate 2>/dev/null || true
python -m pip install -e ".[dev]"

s4dtam-bench validate-readiness configs/readiness/dataset_baseline_matrix.yaml
s4dtam-bench validate-ablation configs/experiments/ablation.yaml
pytest
s4dtam-bench run configs/experiments/smoke.yaml
```

Then work exclusively on the first missing real-data vertical-slice component: the TartanAir raw-to-`sequence.json` converter. Once it exists, the next real-data commands are:

```bash
s4dtam-bench preflight-tartanair data/raw/tartanair-converted

find data/raw/tartanair-converted -name sequence.json -print \
  | sort \
  > artifacts/manifests/tartanair-sequences.txt

sha256sum artifacts/manifests/tartanair-sequences.txt
```

After that, implement and execute the ORB-SLAM3 reproducible runner on those exact frozen sequences. Do not start H1-H7 confirmatory execution before this external vertical slice and the full model freeze are complete.
