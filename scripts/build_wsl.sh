#!/usr/bin/env bash
# Build crane_vision in WSL2 (Ubuntu 22.04).
set -euo pipefail
cd /mnt/d/StudyWorks/3.2/Crane_Extreme
bash scripts/setup_onnxruntime.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
