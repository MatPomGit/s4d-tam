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

This means the final external paper table does **not** need every baseline to run on every dataset. Comparisons should be reported by compatible sensor regime, with unavailable cells reported explicitly rather than filled with an unfair surrogate.

## Metric availability

The readiness file also declares metric-family availability per dataset as `supported`, `derived`, `blocked`, or `unavailable`. A `derived` metric requires a frozen derivation procedure before confirmatory use. For example, a depth-to-occupancy conversion may be scientifically useful, but it must be specified and frozen before occupancy forecasting is evaluated on data that do not directly provide occupancy ground truth.

## Baseline reproduction state

A pinned `revision` or container digest is a specification, not evidence that a baseline has been reproduced. Each mandatory baseline configuration therefore has a `validation` block. It remains `pending_reproduction` until the method has run on its frozen compatible cohort and the normalized result artifact has been checked by the common evaluators. The evidence manifest must then identify the exact input hashes, executable/container identity, hardware policy, output artifacts and validation result.

## TartanAir first vertical slice

TartanAir is the first converter gate because it provides a low-friction visual benchmark for validating coordinate transforms, calibration metadata, trajectory alignment and evaluator behavior.

The strict converted-sequence preflight is:

```bash
s4dtam-bench preflight-tartanair data/raw/tartanair-converted
```

It rejects:

- missing or out-of-order frames;
- missing image files;
- non-finite or non-increasing timestamps;
- incomplete calibration metadata;
- unsupported units;
- malformed positions or quaternions;
- unsupported coordinate-axis conventions.

The CI test suite includes a minimal end-to-end TartanAir adapter fixture so the conversion contract cannot regress silently. This is a **contract vertical slice**, not evidence that a real public sequence has already been validated. Publication readiness still requires running the same gate against pinned upstream files and freezing their checksums.

## Confirmatory freeze gate

The internal H1-H7 study must not begin merely because model files happen to exist. The repository now includes an immutable freeze contract. Start from:

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

The order is:

1. run TartanAir preflight on pinned real sequences and freeze the selected sequence list;
2. reproduce ORB-SLAM3 on those exact visual sequences and normalize the result artifact;
3. validate Blackbird bag synchronization, then reproduce ORB-SLAM3 and VINS-Mono;
4. freeze MARSIM commit/scenarios/seeds, then reproduce FAST-LIO2 and LIO-SAM;
5. audit the actual AeroVerse release before enabling any pair;
6. train and validate the learned S4D-TAM model on train/calibration data only;
7. freeze the learned `full` artifact and H1-H7 artifacts;
8. populate and validate the confirmatory freeze manifest;
9. only then expose the sealed test cohort and execute the external and internal studies as separate analyses.

The learned-model work can proceed in parallel, but the sealed confirmatory test cohort must remain inaccessible until both the data/baseline gates and the model freeze are complete.
