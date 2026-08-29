# Dataset protocol

## Sources and intended use

| Dataset | Primary benchmark role | Important limitation |
|---|---|---|
| [TartanAir](https://tartanair.org/) | visual localization, depth, segmentation, optical flow, adverse appearance | simulated domain; choose fixed environments and difficulty levels |
| [Blackbird UAV Dataset](https://github.com/mit-aera/Blackbird-Dataset) | aggressive visual-inertial flight and motion-capture trajectory | does not provide every semantic or forecasting label |
| [MARSIM](https://github.com/hku-mars/MARSIM) | LiDAR simulation, dynamics, closed-loop stress scenarios | simulator configuration must be archived with every run |
| [AeroVerse](https://arxiv.org/abs/2408.15511) | embodied aerial world-model tasks when the selected release exposes required annotations | verify official files, license, tasks, and version before preregistration |

Never commit vendor data to this repository. Confirm each source license and citation terms.
Blackbird is very large, so download only preregistered trajectories rather than mirroring
the entire collection.

## Normalized dataset layout

```text
data/normalized/<dataset>/
  manifest.json
  sequence_001.npz
  sequence_002.npz
```

Example `manifest.json`:

```json
{
  "dataset_version": "record exact release or checksum",
  "sequences": [
    {
      "id": "sequence_001",
      "file": "sequence_001.npz",
      "occupancy_horizons_s": [1.0],
      "metadata": {"condition": "fog", "license_verified": true}
    }
  ]
}
```

Required NPZ arrays are `timestamps [N]` and `gt_positions [N,3]`. Optional arrays include
`gt_quaternions [N,4]`, `observations [N,D]`, `semantic_observations`, `semantic_gt`,
`occupancy_observations`, and `occupancy_gt_<horizon>`. Ground truth must never be exposed
to an algorithm adapter. Coordinate frames, units, quaternion order, clock correction,
and calibration checks must be documented in the manifest.

## Leakage prevention

- Split by environment or trajectory, not by adjacent frames.
- Freeze test sequence identifiers before final model selection.
- Fit normalization, thresholds, and calibration only on the training/development split.
- Report tuning compute separately from benchmark compute.
- Keep paired seeds and identical sensor-degradation schedules across algorithms.

## Conversion adapters

The adapter kind is selected with `type` in the dataset YAML. Paths, release identifiers,
axis conventions, units, ROS topics, and synchronization tolerances are deliberately explicit;
do not infer these values from filenames. The following Python commands show the reproducible
conversion entry points (replace paths and the pinned version with values from the YAML):

```bash
python -c 'from s4dtam_benchmark.datasets import TartanAirDataset; list(TartanAirDataset("data/raw/tartanair", axis_convention="tartanair_ned_to_enu").sequences())'
python -c 'from s4dtam_benchmark.datasets import BlackbirdDataset; print("configure topics and sync_tolerance_s from configs/datasets/blackbird.yaml")'
python -c 'from s4dtam_benchmark.datasets import MARSIMExporter; MARSIMExporter("data/normalized/marsim", seed=7).export([{"timestamp": 0.0, "position_m": [0, 0, 0]}], simulator_version="PINNED_COMMIT")'
python -c 'from s4dtam_benchmark.datasets import AeroVerseDataset; list(AeroVerseDataset("data/normalized/aeroverse", required_version="paper-release-v1", accepted_license="AeroVerse-release-terms").sequences())'
```

TartanAir consumes a `sequence.json` per trajectory and rejects missing/out-of-order image
indices, absent files or calibration fields, non-increasing time, wrong units, and unknown frame
transforms. Blackbird consumes ROS bags through a configured reader (or dependency-free
`.bag.json` extractions), requires camera/IMU/ground-truth topics, and accepts exactly one match
inside `sync_tolerance_s`. MARSIM sorting and filenames are deterministic for the explicit seed.
AeroVerse produces no partial sequence: its complete manifest, exact release, and explicit
license acceptance must all pass before iteration starts.

## Manifest provenance and license fields

Every generated manifest records `dataset`, `dataset_version`, `timestamp_unit`,
`position_unit`, and `axis_convention`. Simulator exports additionally record `random_seed`.
AeroVerse manifests must contain `"license": {"id": "...", "accepted": true}`. Sequence entries
remain `id`, `file`, optional task horizons, and `metadata`; stable IDs and ordering are part of
the reproducibility contract.

| Dataset | Origin | License/provenance gate |
|---|---|---|
| TartanAir | Carnegie Mellon AirLab synthetic environments (`tartanair.org`) | Record the downloaded release and applicable TartanAir terms. |
| Blackbird | MIT AERA aggressive-flight ROS bags | Record the selected trajectory release and CC BY-NC-SA 3.0 terms. |
| MARSIM | HKU MARS LiDAR simulator | Record the exact simulator commit, GPL-2.0 terms, configuration, and random seed. |
| AeroVerse | AI4CE AeroVerse release | Pin the release identifier and record affirmative acceptance of its release terms before use. |
