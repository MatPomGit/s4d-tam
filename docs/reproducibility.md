# Reproducibility protocol

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
