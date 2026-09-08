#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_URL="https://github.com/UZ-SLAMLab/ORB_SLAM3.git"
UPSTREAM_REV="0df83dde1c85c7ab91a0d47de7a29685d046f637"

usage() {
  cat <<'EOF'
Usage: tools/orb_slam3/build_runner.sh [--source DIR] [--output DIR] [--jobs N]

Builds ORB-SLAM3 v1.0 at the exact pinned commit and adds the S4D-TAM
frame-wise monocular runner. System dependencies (OpenCV, Eigen, Pangolin,
Boost and a C++ compiler) must already be installed.
EOF
}

SOURCE_DIR="artifacts/baselines/orb_slam3/source"
OUTPUT_DIR="artifacts/baselines/orb_slam3"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_DIR="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="$(realpath -m "$SOURCE_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
mkdir -p "$(dirname "$SOURCE_DIR")" "$OUTPUT_DIR/bin"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone "$UPSTREAM_URL" "$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" remote set-url origin "$UPSTREAM_URL"
git -C "$SOURCE_DIR" fetch --tags --force origin
git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_REV"
ACTUAL_REV="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_REV" != "$UPSTREAM_REV" ]]; then
  echo "ORB-SLAM3 revision mismatch: expected $UPSTREAM_REV, got $ACTUAL_REV" >&2
  exit 1
fi

# Build upstream dependencies/library exactly from the pinned source tree.
(
  cd "$SOURCE_DIR"
  chmod +x build.sh
  ./build.sh
)

RUNNER_DST="$SOURCE_DIR/Examples/Monocular/s4dtam_orb_slam3_runner.cc"
cp "$REPO_ROOT/tools/orb_slam3/runner.cpp" "$RUNNER_DST"

MARKER="# S4D-TAM frame-wise runner"
if ! grep -Fq "$MARKER" "$SOURCE_DIR/CMakeLists.txt"; then
  cat >> "$SOURCE_DIR/CMakeLists.txt" <<'EOF'

# S4D-TAM frame-wise runner
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_executable(s4dtam_orb_slam3_runner
        Examples/Monocular/s4dtam_orb_slam3_runner.cc)
target_link_libraries(s4dtam_orb_slam3_runner ${PROJECT_NAME})
EOF
fi

cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SOURCE_DIR/build" --target s4dtam_orb_slam3_runner -j"$JOBS"

RUNNER_BIN="$SOURCE_DIR/Examples/Monocular/s4dtam_orb_slam3_runner"
if [[ ! -x "$RUNNER_BIN" ]]; then
  echo "Runner was not built: $RUNNER_BIN" >&2
  exit 1
fi
cp "$RUNNER_BIN" "$OUTPUT_DIR/bin/s4dtam_orb_slam3_runner"

VOCAB="$SOURCE_DIR/Vocabulary/ORBvoc.txt"
if [[ ! -f "$VOCAB" ]]; then
  echo "ORBvoc.txt not found after upstream build: $VOCAB" >&2
  exit 1
fi
cp "$VOCAB" "$OUTPUT_DIR/ORBvoc.txt"

python3 - "$OUTPUT_DIR/build-manifest.json" "$ACTUAL_REV" "$SOURCE_DIR" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
revision = sys.argv[2]
source = Path(sys.argv[3])
runner = output.parent / "bin" / "s4dtam_orb_slam3_runner"
vocab = output.parent / "ORBvoc.txt"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "schema": "s4dtam-orb-slam3-source-build/v1",
    "upstream": "https://github.com/UZ-SLAMLab/ORB_SLAM3.git",
    "revision": revision,
    "source_path": str(source),
    "runner_sha256": sha256(runner),
    "vocabulary_sha256": sha256(vocab),
    "platform": platform.platform(),
    "compiler": subprocess.run(
        ["c++", "--version"], text=True, stdout=subprocess.PIPE, check=False
    ).stdout.splitlines()[0],
    "cmake": subprocess.run(
        ["cmake", "--version"], text=True, stdout=subprocess.PIPE, check=False
    ).stdout.splitlines()[0],
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Built ORB-SLAM3 runner: $OUTPUT_DIR/bin/s4dtam_orb_slam3_runner"
echo "Vocabulary: $OUTPUT_DIR/ORBvoc.txt"
echo "Build manifest: $OUTPUT_DIR/build-manifest.json"
