#!/usr/bin/env bash
# Download a prebuilt ONNX Runtime (linux-x64) into tools/onnxruntime.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.23.2}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tools/onnxruntime"
ARCH="${ORT_ARCH:-x64}"

if [ -f "${DEST}/lib/libonnxruntime.so" ]; then
  echo "onnxruntime already present at ${DEST}"
  exit 0
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ARCH}-${ORT_VERSION}.tgz"
TMP="$(mktemp -d)"
echo "Downloading ${URL}"
curl -fL --retry 3 -o "${TMP}/ort.tgz" "${URL}"
mkdir -p "${DEST}"
tar -xzf "${TMP}/ort.tgz" -C "${DEST}" --strip-components=1
rm -rf "${TMP}"
echo "onnxruntime ${ORT_VERSION} installed to ${DEST}"
ls "${DEST}/lib"
