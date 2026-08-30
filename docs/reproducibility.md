# Reproducibility protocol

Release metadata is normative: `release/version.yaml` is the single release-version source,
dataset manifests conform to `schemas/dataset-manifest-v1.schema.json`, and every weight bundle
conforms to `schemas/weights-package-v1.schema.json`. Run `s4dtam-release-check` before consuming
or publishing any artifact. The command deliberately fails on version drift, mutable container
references, dependencies without exact versions and SHA-256 hashes, absent licenses, or changed
payloads.

The field-level scientific contract, split-isolation rules, model-state interpretation and
acceptance criteria are specified in [artifact-specification.md](artifact-specification.md).

## Download and verification

1. Download the source archive, its `SHA256SUMS`, all dataset files named by the selected
   manifest, and the selected weight package into a new directory. Do not silently substitute a
   mirror or newer dataset revision.
2. Run `sha256sum --check SHA256SUMS`, then compare every downloaded dataset/calibration file to
   the `sha256` and `bytes` fields in its manifest. Compare the weights, architecture and training
   configuration against `weights/*/metadata.yaml`. A mismatch is a hard failure: delete the file,
   re-download it from the recorded source, and preserve the failure in the run log.
3. Pull containers using the complete `name:version@sha256:digest` reference recorded for the
   release—not `latest`, a branch-like tag, or a tag alone—and verify the resolved digest with the
   runtime's image-inspection command. Docker build inputs use digest-pinned `FROM` images and
   `pip --require-hashes`; never weaken either check.
4. Execute `s4dtam-release-check` at the repository root. This cross-checks `CITATION.cff`, the
   changelog, Python package, container labels/inventory, manifests, weights, licenses and bytes.

Restricted datasets are not redistributed. Obtain them from their owners, retain the original
license/terms, and construct a local manifest containing the exact byte checksums before running.

## Reconstructing a run

Create a clean machine or runner with the platform declared in release metadata. Disable network
access after all verified inputs are staged. Build `containers/benchmark.Dockerfile` (or the
baseline definition), record the resulting image digest, and run the exact tagged configuration,
seed, split and calibration named by the manifest/weights metadata. A typical offline replay is:

```bash
s4dtam-release-check
s4dtam-bench run configs/experiments/offline_benchmark.yaml
```

Capture commit SHA, image digest, command line, environment, hardware/driver versions, input
manifest hashes, stdout/stderr, exit status and output checksums. Repeat stochastic runs according
to the protocol below; do not reuse development calibration results for the test split.

## Before experiments

1. Freeze hypotheses, primary metrics, sequence lists, exclusion rules, seeds, and minimal
   effect sizes in a tagged preregistration file.
2. Record dataset version and file checksums.
3. Pin upstream baseline commits and container digests.
4. Calibrate all methods on development data only.
5. Define one common hardware and power mode for efficiency measurements.

## Execution

- Run at least three warm repetitions for timing and at least five paired mission repeats
  for closed-loop stochastic scenarios; justify final sample size with power or precision.
- Randomize algorithm order where cache, temperature, or battery state can bias results.
- Separate initialization from steady-state latency and report both when relevant.
- Measure end-to-end latency, peak RSS/VRAM, serialized map size, average and peak power,
  and energy per mission.
- Archive configuration, stdout/stderr, failures, system information, and commit SHA.

## Validation stages

1. Offline replay: correctness, localization, semantic and forecast metrics.
2. Software-in-the-loop: closed-loop navigation under controlled GNSS degradation.
3. Hardware-in-the-loop: timing, memory, power, packet loss, and sensor synchronization.
4. Real flight: safety-gated paired missions with an independent abort operator.

## Publication package

Publish the tagged source, environment lock, configuration files, dataset manifests,
normalized non-restricted outputs, per-sequence metrics, statistical scripts, plots, LaTeX
tables, failure log, and a model card. Do not publish restricted dataset content.

The normative hand-off contract is in
[independent-reproduction-package.md](independent-reproduction-package.md), and every
released result must satisfy [compliance-report.md](compliance-report.md).

## Release archive and DOI repository metadata

From a clean signed release commit, run `python tools/build_release.py`. It first runs the release
gate, then creates a deterministic `dist/s4dtam-benchmark-VERSION.tar.gz`, `dist/SHA256SUMS`, and
`dist/repository-metadata.json`, and `dist/release-metadata.json` from tracked files. The latter
explicitly records an unregistered DOI and null output-image digests; replace image nulls only
after building and inspecting the immutable release images. Upload the archive and metadata to a
draft DOI-repository
deposit, verify the repository's rendered metadata and checksum, and only then publish the deposit.
The template intentionally contains no DOI. Record the DOI returned by the repository after actual
registration in a follow-up commit; never predict or reserve a fictitious identifier.

## Retention and disposal

Keep source archives, manifests, licenses, weight metadata, checksum inventories, release logs and
published result tables for at least ten years after the last associated publication. Keep raw
restricted inputs only for the period permitted by their license/consent, encrypted with access
logging; their manifests and non-sensitive provenance remain after deletion. Keep CI scratch files
for 90 days and failed-run diagnostics for one year. Legal, safety, consent and withdrawal duties
override these defaults. Deletion must be documented with artifact identifier, checksum, reason,
authority, date and operator; immutable published releases are superseded, not silently replaced.
