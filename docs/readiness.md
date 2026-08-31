# Dataset and baseline readiness

The external comparison is allowed to use only dataset/baseline pairs whose sensing assumptions are explicitly compatible. Compatibility is not inferred from a method name or from whichever files happen to be present locally.

The executable source of truth is:

```text
configs/readiness/dataset_baseline_matrix.yaml
```

Validate it with:

```bash
s4dtam-bench validate-readiness configs/readiness/dataset_baseline_matrix.yaml
```

A pair has one of three states:

- `supported`: all required sensor streams are declared and the pair has passed its reproducibility gates;
- `blocked`: the sensing assumptions are compatible, but a release, conversion, calibration, environment or provenance gate is still open;
- `not_applicable`: the dataset does not expose the sensor combination required by the baseline and the pair must not enter the external comparison.

`blocked` is intentionally different from `not_applicable`. A blocked pair is planned work. A not-applicable pair should not be made to run by fabricating a missing modality or deriving a sensor stream that the method did not receive in its intended operating regime.

## Current scientific pairing plan

| Dataset | ORB-SLAM3 | VINS-Mono | FAST-LIO2 | LIO-SAM |
| --- | --- | --- | --- | --- |
| TartanAir | blocked, RGB/RGB-D compatible | not applicable without benchmark IMU | not applicable | not applicable |
| Blackbird | blocked, RGB compatible | blocked, RGB+IMU compatible | not applicable | not applicable |
| MARSIM | not applicable | not applicable | blocked, LiDAR+IMU compatible | blocked, LiDAR+IMU compatible |
| AeroVerse | blocked pending release audit | not applicable until IMU is verified | not applicable until LiDAR+IMU is verified | not applicable until LiDAR+IMU is verified |

The final external paper table does **not** need every baseline to run on every dataset. Comparisons are reported by compatible sensor regime, with unavailable cells recorded explicitly rather than filled with an unfair surrogate.

## Metric availability

The readiness file also declares metric-family availability per dataset as `supported`, `derived`, `blocked`, or `unavailable`. A `derived` metric requires a frozen derivation procedure before confirmatory use. For example, a depth-to-occupancy conversion may be scientifically useful, but it must be specified and frozen before occupancy forecasting is evaluated on data that do not directly provide occupancy ground truth.

## Baseline reproduction state

A pinned `revision` or container digest is a specification, not evidence that a baseline has been reproduced. Each mandatory baseline configuration therefore has a `validation` block and remains `pending_reproduction` until the method has actually run on its frozen compatible cohort.

After a real run, validate the complete artifact set with:

```bash
s4dtam-bench validate-baseline-evidence \
  BASELINE \
  DATASET \
  path/to/sequence-list.txt \
  path/to/result-root \
  configs/algorithms/BASELINE.yaml \
  path/to/run-metadata.json \
  path/to/evidence-output
```

The gate requires:

- the exact repository-pinned revision and container;
- the frozen input-manifest SHA-256;
- hardware and exact command metadata from the real execution;
- one normalized result artifact for every sequence in the frozen cohort;
- no extra result artifacts outside that cohort;
- successful parsing of every `s4dtam-algorithm-result-npz/v1` artifact.

The evidence JSON is deterministic for identical inputs so repeating validation does not create a different evidence SHA merely because the validator was run later.

## TartanAir first vertical slice

TartanAir is the first real-data gate because it provides a low-friction visual benchmark for checking coordinate transforms, calibration metadata, trajectory alignment and evaluator behavior.

### Raw V1 ingestion

The executable converter accepts canonical V1-style trajectories containing `pose_left.txt` and `image_left`:

```bash
s4dtam-bench convert-tartanair \
  data/upstream/tartanair \
  data/raw/tartanair-converted \
  --fps VERIFIED_SOURCE_FPS \
  --link-mode symlink
```

The frame rate is intentionally mandatory. The V1 pose file does not carry per-frame timestamps, so the timing assumption must be verified for the selected source material and recorded explicitly rather than hidden as a converter default.

The converter validates frame continuity, pose/image count, finite pose values and unit quaternions. It writes provenance into every `sequence.json` and then loads the whole generated cohort through the production adapter.

### Coordinate convention

TartanAir source poses are represented in NED. The adapter transforms both translation and orientation into ENU. Applying the basis change only to positions would make translational and rotational ground truth internally inconsistent, so this behavior is regression-tested.

### Strict preflight

Run:

```bash
s4dtam-bench preflight-tartanair data/raw/tartanair-converted
```

It rejects:

- missing or out-of-order frames;
- missing image files;
- non-finite or non-increasing timestamps;
- incomplete calibration metadata;
- unsupported units;
- malformed or non-finite positions and quaternions;
- non-unit quaternions;
- unsupported coordinate-axis conventions.

### Cohort freeze

After preflight, freeze the exact converted cohort:

```bash
s4dtam-bench freeze-tartanair \
  data/raw/tartanair-converted \
  artifacts/manifests/tartanair/frozen
```

The command produces:

```text
sequence-list.txt
files.sha256
freeze.json
```

`files.sha256` covers every descriptor and every referenced RGB frame. `freeze.json` records the cohort counts and stable hashes of the file manifest and sequence list.

This completes the software path needed for real TartanAir ingestion, but it is not evidence that public TartanAir sequences have already been processed. The readiness state remains blocked until a pinned real cohort is acquired, converted, validated and frozen.

## Confirmatory freeze gate

The internal H1-H7 study must not begin merely because model files happen to exist. Start from:

```text
configs/reproduction/confirmatory_freeze.template.yaml
```

A completed freeze is accepted only by:

```bash
s4dtam-bench validate-freeze path/to/confirmatory_freeze.yaml
```

The validator requires all of the following at once:

- an exact 40-hex code commit;
- a SHA-256 and external timestamp/registry identifier for the preregistration;
- release identifiers, manifest hashes and sequence-list hashes for all four publication datasets;
- validated evidence manifests and configuration hashes for all four external baselines;
- immutable artifact and configuration hashes for `full` plus every H1-H7 model;
- the frozen seed list;
- Holm multiplicity correction, 10,000 bootstrap resamples and the 2 percentage-point H7 non-inferiority margin.

The supplied template intentionally has `study_state: draft` and cannot pass the validator. This prevents accidental execution of a confirmatory study from a partially populated manifest.

## Next executable gates

The order is now:

1. acquire a pinned real TartanAir V1 cohort and verify its sampling rate;
2. run `convert-tartanair`, `preflight-tartanair` and `freeze-tartanair`;
3. reproduce ORB-SLAM3 on exactly the frozen sequence list;
4. run `validate-baseline-evidence` for the TartanAir/ORB-SLAM3 cohort;
5. run the first TartanAir-only development comparison through the common evaluator;
6. validate Blackbird bag synchronization, then reproduce ORB-SLAM3 and VINS-Mono;
7. freeze MARSIM commit/scenarios/seeds, then reproduce FAST-LIO2 and LIO-SAM;
8. audit the actual AeroVerse release before enabling any pair;
9. train and validate the learned S4D-TAM model on train/calibration data only;
10. freeze the learned `full` artifact and H1-H7 artifacts;
11. populate and validate the confirmatory freeze manifest;
12. only then expose the sealed test cohort and execute the external and internal studies as separate analyses.

The learned-model work may proceed in parallel, but the sealed confirmatory test cohort must remain inaccessible until both the data/baseline gates and the model freeze are complete.
