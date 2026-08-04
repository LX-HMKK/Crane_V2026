#pragma once

#include <map>
#include <optional>
#include <set>
#include <utility>

#include "common/types.hpp"

namespace crane {

// Logical completion of missing/duplicate/ambiguous per-zone detections.
// Faithful port of models/logic_completion.py.
//
// zone_vals: zone_id -> (value, confidence), or nullopt for no detection.
// required_set: values that must appear exactly once across all zones.
//
// Returns zone_id -> ZoneResult (value, inferred flag, confidence).
// Throws std::invalid_argument when zone count != required count.
//
// Note: confidence_threshold is accepted for API parity with the Python
// version, where it is likewise unused — low-confidence detections still
// participate in scoring.
std::map<int, ZoneResult> validate_and_complete(
    const std::map<int, std::optional<std::pair<int, float>>>& zone_vals,
    const std::set<int>& required_set, float confidence_threshold = 0.5f);

int count_inferred(const std::map<int, ZoneResult>& results);

}  // namespace crane
