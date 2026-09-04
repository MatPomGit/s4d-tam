# Configuration paths

## One path-resolution rule

Every relative path read from an experiment YAML is resolved against the directory that
contains that YAML file, **not** against the process working directory. This contract applies
to dataset `root` and `manifest`, algorithm `reference_map` and `result_root`, and `output_dir`.
It therefore remains safe to invoke the same configuration from the repository root, a job
scheduler working directory, or any other directory.

`output_dir` deliberately follows the same rule. If it is omitted, the default `outputs/run`
is created below the configuration directory. CLI positional paths used by conversion,
preflight, freezing, evidence, and package-verification commands are ordinary shell paths and
remain relative to the caller's current directory; they are not values read from YAML.

The run manifest preserves the YAML values under `config` and records path provenance under
`path_resolution`. Each entry contains both `provided` (the value authored by the user)
and `resolved` (the absolute path passed to an adapter).

## Relative paths

For `/work/study/experiment.yaml`:

```yaml
output_dir: artifacts/run-01
datasets:
  - type: manifest
    name: flight
    root: data/normalized
    manifest: manifests/flight.json
algorithms:
  - type: external_artifact
    name: orb_slam3
    result_root: baseline-results/orb_slam3
```

the output directory is `/work/study/artifacts/run-01`, regardless of the invocation CWD.
The manifest path is `/work/study/manifests/flight.json`; it does not need to be inside `root`.

## Absolute paths and optional paths

Absolute paths are accepted unchanged (after normalizing them to an absolute canonical form):

```yaml
output_dir: /mnt/results/run-01
datasets:
  - type: marsim
    root: /mnt/datasets/marsim
    manifest: null
algorithms:
  - type: s4dtam_reference
    name: s4d_tam_reference
    reference_map: /mnt/maps/site-a.json
```

Optional path values such as `manifest: null` and `reference_map: null` remain `null`; they do
not resolve to the configuration directory or to a path named `None`.
