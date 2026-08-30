# Scientific artifact and release specification

## Scope and evidential status

This document defines the minimum evidence needed to independently reconstruct an S4D-TAM
benchmark result. A valid checksum establishes byte identity, not scientific validity. A successful
release check therefore does not imply that a dataset is representative, a calibration is unbiased,
or a model generalises beyond its stated population. Those claims require the preregistered design,
uncertainty analysis and independent replication described in the methodology documents.

The release unit consists of a signed source tag, immutable data manifests, immutable weight
packages, digest-addressed execution environments, experiment configurations and machine-readable
results. `release/version.yaml` is the release identity; schema versions evolve independently so a
compatible schema revision need not imply a new scientific result.

## Dataset manifest methodology

Each manifest identifies the target population, sampling process, modalities, coordinate-frame
conventions, inclusion and exclusion criteria, known limitations and leakage policy. Split members
are sequence identifiers rather than implicit filesystem globs. This prevents a later download from
silently changing the experimental unit. The following rules are normative:

1. Assign an acquisition unit (flight, scene or independently generated simulation seed) to exactly
   one split before fitting, calibration or threshold selection. Correlated frames from one unit
   must never cross splits.
2. Use `training` only to estimate model parameters, `validation` to select hyperparameters and
   stopping rules, `calibration` to estimate uncertainty-calibration parameters, and `test` once for
   the preregistered final analysis. Report any departure as a protocol deviation.
3. Record all exclusions before inspecting method-specific test errors. Preserve excluded sequence
   identifiers and reason codes so denominators can be audited.
4. Authenticate every payload and calibration file independently. The `bytes` field detects common
   truncation errors before hashing; SHA-256 establishes exact identity.
5. Record sensor units, axes, handedness, clock domain and nominal sampling rate. Conversion must be
   a deterministic command associated with an immutable source commit and container digest.

The repository fixture is intentionally non-scientific. It verifies the contract but is too small
to estimate accuracy, uncertainty or external validity; its manifest states that limitation.

## Weights-package methodology

A weight package binds parameters to the only architecture configuration that can interpret them,
the training configuration and source commit that produced them, and all contributing dataset
manifests. Data roles distinguish estimation, model selection, calibration and testing. The package
also records framework ABI, preprocessing, seed, determinism limitations, hardware and stopping
rule. These fields make numerical discrepancies diagnosable rather than merely observable.

Calibration is part of the model state. Record the objective, fitted split, sample count and all
learned parameters. Calibration data must be disjoint from final test units. Report both predictive
performance and calibration metrics with units, aggregation level, confidence-interval method and
split. Do not select a checkpoint from final-test performance.

Every package must state intended uses and known limitations. Loading a package outside the declared
sensor contract, coordinate convention, population or software ABI is an extrapolation and must be
reported as such. The fixture package contains no learned model and must never appear in a benchmark
comparison.

## Container and dependency provenance

Dockerfiles pin `FROM` by tag and OCI digest. Python requirements pin the normalized distribution
name and version and authorize only artifacts whose SHA-256 appears in `requirements.lock`.
Release images are built without network access after the base image and wheelhouse have been
verified. The resulting benchmark and baseline image digests are written into the final release
metadata after building; they cannot be known honestly in a source template.

For heterogeneous baselines, preserve upstream commit, patches, compiler and accelerator ABI in a
separate image. Do not install baseline dependencies into the benchmark image. Compare methods on
the common input/output contract and hardware power mode, not by forcing incompatible runtimes into
one environment.

## Replication design and acceptance

An independent replicator starts from the archive, not a maintainer workspace. Verify the archive,
materialize inputs from manifests, rebuild images, then execute the preregistered sequence order and
seeds. Record warm-up policy, repetitions, failures, hardware, driver, image and input digests.
Primary estimates use the acquisition unit—not individual frames—as the independent unit. Report
paired per-sequence differences, effect sizes and interval estimates; multiple secondary hypotheses
use the preregistered correction. Failed missions remain in the denominator under the declared
failure rule.

Replication is exact when inputs and environment digests match and outputs are byte-identical.
Where nondeterministic accelerator kernels preclude exact equality, use the absolute/relative
tolerances declared before execution and compare scientific conclusions as well as point values.
A discrepancy report must retain both outputs, logs, environment inventories and a minimal failing
sequence. Never widen tolerances after seeing the discrepancy.

## Release state transition

The source archive is generated only from a clean, tagged commit. A draft repository deposit may be
updated until verification. Publication makes the deposit immutable and returns the DOI; only that
returned identifier may then be added to citation metadata. Corrections create a new release and
explicitly relate it to the superseded record rather than overwriting evidence.
