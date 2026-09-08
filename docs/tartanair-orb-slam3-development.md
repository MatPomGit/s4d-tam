# TartanAir + ORB-SLAM3 development vertical slice

This runbook defines the first end-to-end development comparison used to verify the S4D-TAM publication pipeline before the confirmatory study is frozen.

It is deliberately restricted to one dataset and one independent external baseline:

- dataset: TartanAir V1-compatible converted cohort;
- candidate: `s4d_tam_reference`;
- baseline: ORB-SLAM3;
- phase: `development`;
- experiment: `configs/experiments/tartanair_orb_slam3_development.yaml`.

Results from this run are engineering/development evidence only. They must not be reported as confirmatory hypothesis tests.

## 1. Repository health

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
s4dtam-bench validate-comparison configs/experiments/tartanair_orb_slam3_development.yaml
```

## 2. Convert and preflight the selected TartanAir cohort

Set the exact upstream location and a sampling frequency verified for that release:

```bash
export TARTANAIR_SOURCE=/absolute/path/to/tartanair-v1
export TARTANAIR_FPS=<VERIFIED_FPS>

s4dtam-bench convert-tartanair \
  "$TARTANAIR_SOURCE" \
  data/raw/tartanair-converted \
  --fps "$TARTANAIR_FPS" \
  --link-mode symlink

s4dtam-bench preflight-tartanair data/raw/tartanair-converted
```

Do not guess `TARTANAIR_FPS`; the V1 `pose_left.txt` source does not carry timestamps and the declared sampling rate enters the trajectory time base.

## 3. Freeze the cohort

```bash
mkdir -p artifacts/manifests/tartanair
s4dtam-bench freeze-tartanair \
  data/raw/tartanair-converted \
  artifacts/manifests/tartanair/frozen

cat artifacts/manifests/tartanair/frozen/sequence-list.txt
cat artifacts/manifests/tartanair/frozen/freeze.json
```

The ORB-SLAM3 run and S4D-TAM evaluation must use exactly this frozen sequence list.

## 4. Reproduce the pinned ORB-SLAM3 implementation

The baseline specification is `configs/algorithms/orb_slam3.yaml`. The repository pins both upstream revision and container digest. Do not substitute another revision or a mutable image tag.

```bash
python - <<'PY'
from s4dtam_benchmark.config import load_yaml
cfg = load_yaml("configs/algorithms/orb_slam3.yaml")
print("upstream:", cfg["upstream"])
print("revision:", cfg["revision"])
print("container:", cfg["container"])
PY
```

The official ORB-SLAM3 interface requires a calibrated settings file and vocabulary file for the selected camera mode. The TartanAir adapter currently exposes monocular RGB observations and camera calibration through each `sequence.json`; therefore the reproduction wrapper must preserve those parameters instead of reusing an unrelated TUM/EuRoC calibration.

Expected normalized outputs:

```text
outputs/baselines/orb_slam3/tartanair/<sequence-id>.npz
```

Each output must satisfy the repository `s4dtam-algorithm-result-npz/v1` contract and include trajectory timing plus recorded resource information required by the baseline-evidence validator.

## 5. Freeze baseline evidence

Create run metadata containing the actual hardware and exact executed command, then validate the complete frozen cohort:

```bash
s4dtam-bench validate-baseline-evidence \
  orb_slam3 \
  tartanair \
  artifacts/manifests/tartanair/frozen/sequence-list.txt \
  outputs/baselines/orb_slam3 \
  configs/algorithms/orb_slam3.yaml \
  artifacts/baselines/orb-slam3-tartanair-run.json \
  artifacts/baselines/evidence
```

This gate must fail if any frozen sequence is missing, an unexpected output appears, the implementation is not pinned, or run metadata are incomplete.

## 6. Execute the common comparison

Only after baseline evidence validates:

```bash
s4dtam-bench validate-comparison \
  configs/experiments/tartanair_orb_slam3_development.yaml

s4dtam-bench run \
  configs/experiments/tartanair_orb_slam3_development.yaml
```

Expected publication-pipeline output root:

```text
outputs/tartanair_orb_slam3_development/
```

Both systems are evaluated through the same evaluator and reporting path.

## 7. Development acceptance criteria

The vertical slice is complete only when all of the following are true:

1. the TartanAir cohort passes conversion and strict preflight;
2. its sequence list and files are frozen with SHA-256 evidence;
3. the pinned ORB-SLAM3 implementation has produced one valid normalized artifact for every frozen sequence;
4. `validate-baseline-evidence` passes;
5. `validate-comparison` passes for the dedicated development configuration;
6. the common evaluator completes without silent metric imputation;
7. generated tables/plots can be traced to immutable input and baseline evidence.

Completion of this run does not unlock confirmatory H1-H7 testing. The full confirmatory external matrix, learned-model freeze, preregistered ablations and field/field-analog validation remain separate publication gates.
