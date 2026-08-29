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
