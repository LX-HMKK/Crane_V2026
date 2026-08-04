#include "task_generator.hpp"

#include <cmath>
#include <set>

#include "logic_completion.hpp"
#include "mapping.hpp"

namespace crane {

TaskGenerator::TaskGenerator(std::map<std::string, int> mapping,
                             LinesConfig lines,
                             std::map<int, std::string> bean_names,
                             std::map<int, std::string> digit_names)
    : mapping_(std::move(mapping)),
      lines_(lines),
      bean_names_(std::move(bean_names)),
      digit_names_(std::move(digit_names)) {}

std::string TaskGenerator::validate_lines() const {
  const auto& l = lines_;
  auto is_num = [](double v) { return !std::isnan(v) && !std::isinf(v); };
  if (!is_num(l.bean_left_x) || !is_num(l.bean_right_x)) {
    return "invalid bean line coordinate";
  }
  if (!is_num(l.digit_x1) || !is_num(l.digit_x2) || !is_num(l.digit_x3) ||
      !is_num(l.digit_x4)) {
    return "invalid digit line coordinate";
  }
  if (l.bean_left_x < 0 || l.bean_right_x < 0) {
    return "bean line coordinates must be non-negative";
  }
  if (l.digit_x1 < 0 || l.digit_x2 < 0 || l.digit_x3 < 0 || l.digit_x4 < 0) {
    return "digit line coordinates must be non-negative";
  }
  if (l.bean_left_x >= l.bean_right_x) {
    return "bean left_x must be less than right_x";
  }
  if (!(l.digit_x1 < l.digit_x2 && l.digit_x2 < l.digit_x3 &&
        l.digit_x3 < l.digit_x4)) {
    return "digit x1..x4 must be strictly increasing";
  }
  return "";
}

int TaskGenerator::bean_zone(float cx) const {
  // Strict < is intentional: detections exactly on a boundary line fall
  // into the zone to the right of that line.
  if (cx < lines_.bean_left_x) return 1;
  if (cx < lines_.bean_right_x) return 2;
  return 3;
}

int TaskGenerator::digit_zone(float cx) const {
  if (cx < lines_.digit_x1) return 4;
  if (cx < lines_.digit_x2) return 5;
  if (cx < lines_.digit_x3) return 6;
  if (cx < lines_.digit_x4) return 7;
  return 8;
}

int TaskGenerator::bean_color_code(const std::string& name) const {
  return bean_name_to_digit(name, mapping_);
}

std::map<int, std::optional<Detection>> TaskGenerator::aggregate_beans(
    const Detections& dets) const {
  std::map<int, std::optional<Detection>> best = {
      {1, std::nullopt}, {2, std::nullopt}, {3, std::nullopt}};
  for (const auto& det : dets) {
    int zone = bean_zone(det.cx());
    auto& cur = best[zone];
    if (!cur.has_value() || det.conf > cur->conf) cur = det;
  }
  return best;
}

std::map<int, std::optional<Detection>> TaskGenerator::aggregate_digits(
    const Detections& dets) const {
  std::map<int, std::optional<Detection>> best = {{4, std::nullopt},
                                                  {5, std::nullopt},
                                                  {6, std::nullopt},
                                                  {7, std::nullopt},
                                                  {8, std::nullopt}};
  for (const auto& det : dets) {
    int zone = digit_zone(det.cx());
    auto& cur = best[zone];
    if (!cur.has_value() || det.conf > cur->conf) cur = det;
  }
  return best;
}

TaskGenerator::CompletionResult TaskGenerator::generate_frame_with_completion(
    const Detections& bean_dets, const Detections& digit_dets,
    float confidence_threshold, int max_inferred_beans,
    int max_inferred_digits) const {
  CompletionResult out;
  std::string err = validate_lines();
  if (!err.empty()) {
    out.reason = err;
    return out;
  }

  auto bean_best = aggregate_beans(bean_dets);
  auto digit_best = aggregate_digits(digit_dets);

  // ── map bean zones to (color_code, confidence) ─────────────────────
  std::map<int, std::optional<std::pair<int, float>>> bean_zone_vals;
  for (int zone = 1; zone <= 3; ++zone) {
    bean_zone_vals[zone] = std::nullopt;
    const auto& det = bean_best[zone];
    if (!det.has_value()) continue;
    auto name_it = bean_names_.find(det->cls_id);
    if (name_it == bean_names_.end()) continue;
    int code = bean_color_code(name_it->second);
    if (code < 0) continue;
    bean_zone_vals[zone] = std::make_pair(code, det->conf);
  }

  // ── map digit zones to (digit_value, confidence) ───────────────────
  std::map<int, std::optional<std::pair<int, float>>> digit_zone_vals;
  for (int zone = 4; zone <= 8; ++zone) {
    digit_zone_vals[zone] = std::nullopt;
    const auto& det = digit_best[zone];
    if (!det.has_value()) continue;
    auto name_it = digit_names_.find(det->cls_id);
    if (name_it == digit_names_.end()) continue;
    int val;
    try {
      val = std::stoi(name_it->second);
    } catch (...) {
      continue;
    }
    if (val < 1 || val > 5) continue;
    digit_zone_vals[zone] = std::make_pair(val, det->conf);
  }

  // ── logical completion ─────────────────────────────────────────────
  out.bean_results = validate_and_complete(bean_zone_vals, {1, 2, 3},
                                           confidence_threshold);
  out.digit_results = validate_and_complete(digit_zone_vals,
                                            {1, 2, 3, 4, 5},
                                            confidence_threshold);

  // ── guard: too many inferred zones → reject ────────────────────────
  int bean_inferred = count_inferred(out.bean_results);
  int digit_inferred = count_inferred(out.digit_results);
  if (bean_inferred > max_inferred_beans) {
    out.reason = "too many inferred bean zones (" +
                 std::to_string(bean_inferred) + " > " +
                 std::to_string(max_inferred_beans) + ")";
    return out;
  }
  if (digit_inferred > max_inferred_digits) {
    out.reason = "too many inferred digit zones (" +
                 std::to_string(digit_inferred) + " > " +
                 std::to_string(max_inferred_digits) + ")";
    return out;
  }

  FramePayload payload = {static_cast<uint8_t>(out.bean_results[1].value),
                          static_cast<uint8_t>(out.bean_results[2].value),
                          static_cast<uint8_t>(out.bean_results[3].value),
                          static_cast<uint8_t>(out.digit_results[4].value),
                          static_cast<uint8_t>(out.digit_results[5].value),
                          static_cast<uint8_t>(out.digit_results[6].value),
                          static_cast<uint8_t>(out.digit_results[7].value),
                          static_cast<uint8_t>(out.digit_results[8].value)};
  out.payload = payload;
  out.reason = "ok";
  return out;
}

std::pair<std::optional<FramePayload>, std::string>
TaskGenerator::generate_frame(const Detections& bean_dets,
                              const Detections& digit_dets) const {
  std::string err = validate_lines();
  if (!err.empty()) return {std::nullopt, err};

  auto bean_best = aggregate_beans(bean_dets);

  std::map<int, int> colors;
  for (int zone = 1; zone <= 3; ++zone) {
    const auto& det = bean_best[zone];
    if (!det.has_value()) {
      return {std::nullopt,
              "missing bean in pickup zone " + std::to_string(zone)};
    }
    auto name_it = bean_names_.find(det->cls_id);
    if (name_it == bean_names_.end()) {
      return {std::nullopt, "unknown bean class " +
                                std::to_string(det->cls_id) + " in zone " +
                                std::to_string(zone)};
    }
    int code = bean_color_code(name_it->second);
    if (code < 0) {
      return {std::nullopt, "unmapped bean class '" + name_it->second +
                                "' in zone " + std::to_string(zone)};
    }
    colors[zone] = code;
  }
  {
    std::set<int> vals;
    for (const auto& kv : colors) vals.insert(kv.second);
    if (vals != std::set<int>{1, 2, 3} || colors.size() != 3) {
      return {std::nullopt, "pickup colors must be 1,2,3 exactly once"};
    }
  }

  auto digit_best = aggregate_digits(digit_dets);
  std::map<int, int> digits;
  for (int zone = 4; zone <= 8; ++zone) {
    const auto& det = digit_best[zone];
    if (!det.has_value()) {
      return {std::nullopt,
              "missing digit in placement zone " + std::to_string(zone)};
    }
    auto name_it = digit_names_.find(det->cls_id);
    if (name_it == digit_names_.end()) {
      return {std::nullopt, "unknown digit class " +
                                std::to_string(det->cls_id) + " in zone " +
                                std::to_string(zone)};
    }
    int value;
    try {
      value = std::stoi(name_it->second);
    } catch (...) {
      return {std::nullopt, "non-numeric digit name '" + name_it->second +
                                "' in zone " + std::to_string(zone)};
    }
    if (value < 1 || value > 5) {
      return {std::nullopt, "digit value " + std::to_string(value) +
                                " in zone " + std::to_string(zone) +
                                " out of range 1..5"};
    }
    digits[zone] = value;
  }
  {
    std::set<int> vals;
    for (const auto& kv : digits) vals.insert(kv.second);
    if (vals != std::set<int>{1, 2, 3, 4, 5} || digits.size() != 5) {
      return {std::nullopt, "placement digits must be 1..5 exactly once"};
    }
  }

  FramePayload payload = {static_cast<uint8_t>(colors[1]),
                          static_cast<uint8_t>(colors[2]),
                          static_cast<uint8_t>(colors[3]),
                          static_cast<uint8_t>(digits[4]),
                          static_cast<uint8_t>(digits[5]),
                          static_cast<uint8_t>(digits[6]),
                          static_cast<uint8_t>(digits[7]),
                          static_cast<uint8_t>(digits[8])};
  return {payload, "ok"};
}

}  // namespace crane
