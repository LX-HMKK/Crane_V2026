#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>

#include "common/types.hpp"

namespace crane {

// Vertical guide-line configuration dividing pickup zones 1-3 (bean camera)
// and placement zones 4-8 (digit camera). NaN marks a missing/invalid value
// (mirrors malformed-YAML handling in the Python _validate_lines).
struct LinesConfig {
  double bean_left_x = 0.0 / 0.0;
  double bean_right_x = 0.0 / 0.0;
  double digit_x1 = 0.0 / 0.0;
  double digit_x2 = 0.0 / 0.0;
  double digit_x3 = 0.0 / 0.0;
  double digit_x4 = 0.0 / 0.0;
};

// Convert bean/digit detections into result-frame payloads.
// Faithful port of models/task_generator.py.
class TaskGenerator {
 public:
  TaskGenerator(std::map<std::string, int> mapping, LinesConfig lines,
                std::map<int, std::string> bean_names,
                std::map<int, std::string> digit_names);

  // Zone assignment by detection center. Boundary goes to the right zone.
  int bean_zone(float cx) const;   // 1..3
  int digit_zone(float cx) const;  // 4..8

  struct CompletionResult {
    std::optional<FramePayload> payload;
    std::map<int, ZoneResult> bean_results;
    std::map<int, ZoneResult> digit_results;
    std::string reason;
  };

  // Tolerant variant: infers missing/duplicate zones via confidence-weighted
  // permutation scoring; rejects when inferred counts exceed the limits.
  CompletionResult generate_frame_with_completion(
      const Detections& bean_dets, const Detections& digit_dets,
      float confidence_threshold = 0.5f, int max_inferred_beans = 1,
      int max_inferred_digits = 2) const;

  // Strict variant: every zone must have a detection, colors/digits unique.
  std::pair<std::optional<FramePayload>, std::string> generate_frame(
      const Detections& bean_dets, const Detections& digit_dets) const;

  const std::map<int, std::string>& bean_names() const { return bean_names_; }
  const std::map<int, std::string>& digit_names() const { return digit_names_; }
  const std::map<std::string, int>& mapping() const { return mapping_; }
  const LinesConfig& lines() const { return lines_; }

 private:
  // Returns empty string when valid, else the reason (Python _validate_lines).
  std::string validate_lines() const;
  // zone -> best detection (highest conf); every zone always present.
  std::map<int, std::optional<Detection>> aggregate_beans(
      const Detections& dets) const;
  std::map<int, std::optional<Detection>> aggregate_digits(
      const Detections& dets) const;
  int bean_color_code(const std::string& name) const;  // -1 when unmapped

  std::map<std::string, int> mapping_;
  LinesConfig lines_;
  std::map<int, std::string> bean_names_;
  std::map<int, std::string> digit_names_;
};

}  // namespace crane
