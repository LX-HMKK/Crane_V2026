#!/usr/bin/env bash
# Build crane_vision on the RDK X5 (arm64) with the real BPU backend.
# Run ON THE BOARD, from the project root.
set -euo pipefail

# 1. Dependencies (RDK X5 stock image already has libdnn in /usr/lib/hbbpu).
if ! pkg-config --exists opencv4 2>/dev/null; then
  echo "Installing OpenCV + yaml-cpp..."
  sudo apt-get update
  sudo apt-get install -y libopencv-dev libyaml-cpp-dev cmake g++
fi

# 2. ONNX Runtime is not needed on the board (BPU backend); disable it to
#    skip the download. Set CRANE_WITH_ONNXRUNTIME=ON if you want to compare.
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRANE_WITH_ONNXRUNTIME=OFF \
  -DCRANE_WITH_BPU=ON
cmake --build build -j"$(nproc)"

echo
echo "Built ./build/crane_vision (BPU backend: $([ -e /usr/lib/hbbpu/libdnn.so ] && echo real libDNN || echo MISSING — check /usr/lib/hbbpu))"
echo "Run:  sudo ./build/crane_vision --backend bpu --web-port 8080"
