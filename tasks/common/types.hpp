#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace crane {

// One object detection in original-image coordinates.
// Mirrors the Python [x1, y1, x2, y2, conf, cls_id] list.
struct Detection {
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  float conf = 0;
  int cls_id = -1;

  float cx() const { return (x1 + x2) * 0.5f; }
  float cy() const { return (y1 + y2) * 0.5f; }
};

using Detections = std::vector<Detection>;

// Result frame payload: C1 C2 C3 N4 N5 N6 N7 N8.
using FramePayload = std::array<uint8_t, 8>;

// Per-zone completion result: value, source ("detected"/"inferred"), conf.
struct ZoneResult {
  int value = 0;
  bool inferred = false;
  std::optional<float> conf;
};

}  // namespace crane
