#!/usr/bin/env bash
# Run crane_vision (or ctest) inside the project directory with the ORT
# library path set. Usage: bash scripts/wsl_run.sh ./build/crane_vision --once ...
set -euo pipefail
cd /mnt/d/StudyWorks/3.2/Crane_Extreme
export LD_LIBRARY_PATH="${PWD}/build:${PWD}/tools/onnxruntime/lib:${LD_LIBRARY_PATH:-}"
exec "$@"
