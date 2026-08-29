# Metric specification

## Primary endpoints

Use a small preregistered primary set to avoid selective reporting:

- localization: ATE RMSE after SE(3) alignment without scale;
- forecasting: occupancy IoU at 1 s and 3 s, plus Brier score or ECE;
- safety: mission success and collisions per kilometre;
- efficiency: p95 latency and map-memory footprint.

Report all secondary metrics, even when a primary endpoint is not significant.

## Definitions

| Metric | Direction | Unit / aggregation |
|---|---:|---|
| ATE RMSE | lower | metres, sequence level |
| RPE translation RMSE | lower | metres per fixed frame interval |
| final drift | lower | metres and percent of reference path length |
| mIoU / macro F1 | higher | class-balanced |
| occupancy IoU / F1 | higher | each forecast horizon separately |
| Brier / NLL / ECE | lower | probabilistic calibration |
| pose NEES / 95% coverage / NLL | calibrated | consistency of predicted pose covariance |
| risk AUROC / false alarm / miss rate | higher / lower | safety-relevant event prediction |
| flow EPE | lower | metric units defined by exported grid |
| mission success | higher | proportion of repeated missions |
| collisions/km | lower | closed-loop missions only |
| path efficiency | higher | shortest feasible path / executed path |
| latency p95 | lower | milliseconds, include preprocessing and map update |
| map bytes / peak RSS | lower | bytes / MiB |

Rotation RPE should be added when orientation ground truth and prediction are both
available. For loop closure and relocalization, additionally record success rate, time to
relocalize, and post-relocalization pose error.

## Statistical unit

The independent unit is a sequence, mission, or environment-level repeat, never an image
frame. Use paired comparisons under identical seeds and conditions. Report paired mean or
median differences, 95% confidence intervals, standardized effect size, sample count, and
Holm-adjusted p-values for families of confirmatory tests.

## Unavailable metrics

The runner writes `unavailable_metrics.json`. A metric is unavailable when either the
dataset lacks ground truth or the algorithm does not emit the prediction. This is distinct
from failure, which is written to `failures.json`.
