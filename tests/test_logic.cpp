// Logic completion + task generator tests, ported from
// crane/tests/test_logic_completion.py (same vectors, same expectations).
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <string>

#include "logic/logic_completion.hpp"
#include "logic/mapping.hpp"
#include "logic/task_generator.hpp"

namespace {

int g_pass = 0, g_fail = 0;

#define CHECK(cond, ...)                              \
  do {                                                \
    if (cond) {                                       \
      ++g_pass;                                       \
    } else {                                          \
      ++g_fail;                                       \
      std::printf("FAIL %s:%d: ", __func__, __LINE__); \
      std::printf(__VA_ARGS__);                       \
      std::printf("\n");                              \
    }                                                 \
  } while (0)

using crane::Detection;
using crane::Detections;
using crane::TaskGenerator;

crane::LinesConfig make_lines() {
  crane::LinesConfig l;
  l.bean_left_x = 500;
  l.bean_right_x = 800;
  l.digit_x1 = 200;
  l.digit_x2 = 400;
  l.digit_x3 = 600;
  l.digit_x4 = 800;
  return l;
}

TaskGenerator make_tg() {
  return TaskGenerator(crane::kDefaultBeanToDigit, make_lines(),
                       {{0, "yellow"}, {1, "green"}, {2, "white"}},
                       {{0, "1"}, {1, "2"}, {2, "3"}, {3, "4"}, {4, "5"}});
}

Detection det(int cls_id, float cx, float cy = 360, float conf = 0.9f) {
  Detection d;
  d.x1 = cx - 30;
  d.y1 = cy - 30;
  d.x2 = cx + 30;
  d.y2 = cy + 30;
  d.conf = conf;
  d.cls_id = cls_id;
  return d;
}

using crane::FramePayload;

void test_all_detected() {
  auto tg = make_tg();
  Detections beans = {det(0, 300), det(1, 600), det(2, 900)};
  Detections digits = {det(0, 100), det(1, 300), det(2, 500), det(3, 700),
                       det(4, 900)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(r.payload.has_value(), "payload missing: %s", r.reason.c_str());
  CHECK((*r.payload == FramePayload{1, 2, 3, 1, 2, 3, 4, 5}), "wrong payload");
  CHECK(crane::count_inferred(r.bean_results) == 0, "bean inferred != 0");
  CHECK(crane::count_inferred(r.digit_results) == 0, "digit inferred != 0");
}

void test_one_bean_missing() {
  auto tg = make_tg();
  Detections beans = {det(0, 300, 360, 0.9f), det(2, 900, 360, 0.8f)};
  Detections digits = {det(0, 100), det(1, 300), det(2, 500), det(3, 700),
                       det(4, 900)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(r.payload.has_value(), "payload missing: %s", r.reason.c_str());
  CHECK((*r.payload)[0] == 1 && (*r.payload)[1] == 2 && (*r.payload)[2] == 3,
        "wrong C1-C3");
  CHECK(r.bean_results[2].inferred, "Z2 should be inferred");
  CHECK(!r.bean_results[1].inferred, "Z1 should be detected");
}

void test_one_digit_missing() {
  auto tg = make_tg();
  Detections beans = {det(0, 300), det(1, 600), det(2, 900)};
  Detections digits = {det(0, 100), det(1, 300), det(3, 700), det(4, 900)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(r.payload.has_value(), "payload missing: %s", r.reason.c_str());
  FramePayload want{1, 2, 3, 1, 2, 3, 4, 5};
  CHECK((*r.payload == want), "wrong N4-N8");
  CHECK(r.digit_results[6].inferred, "Z6 should be inferred");
}

void test_bean_duplicate_resolved() {
  auto tg = make_tg();
  Detections beans = {det(0, 300, 360, 0.55f), det(0, 600, 360, 0.90f),
                      det(2, 900, 360, 0.80f)};
  Detections digits = {det(0, 100), det(1, 300), det(2, 500), det(3, 700),
                       det(4, 900)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(r.payload.has_value(), "payload missing: %s", r.reason.c_str());
  CHECK(r.bean_results[2].value == 1 && !r.bean_results[2].inferred,
        "Z2 should get value 1 detected");
  CHECK(r.bean_results[1].value == 2 && r.bean_results[1].inferred,
        "Z1 should get value 2 inferred");
}

void test_too_many_bean_missing() {
  auto tg = make_tg();
  Detections beans = {det(0, 300, 360, 0.9f)};
  Detections digits = {det(0, 100), det(1, 300), det(2, 500), det(3, 700),
                       det(4, 900)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(!r.payload.has_value(), "payload should be rejected");
  CHECK(r.reason.find("inferred bean") != std::string::npos,
        "reason should mention inferred bean: %s", r.reason.c_str());
}

void test_too_many_digit_missing() {
  auto tg = make_tg();
  Detections beans = {det(0, 300), det(1, 600), det(2, 900)};
  Detections digits = {det(0, 100), det(1, 300)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(!r.payload.has_value(), "payload should be rejected");
  CHECK(r.reason.find("inferred digit") != std::string::npos,
        "reason should mention inferred digit: %s", r.reason.c_str());
}

void test_unknown_bean_class_ignored() {
  auto tg = make_tg();
  Detections beans = {det(99, 300, 360, 0.9f), det(1, 600, 360, 0.9f),
                      det(2, 900, 360, 0.9f)};
  Detections digits = {det(0, 100), det(1, 300), det(2, 500), det(3, 700),
                       det(4, 900)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(r.payload.has_value(), "payload missing: %s", r.reason.c_str());
  CHECK(r.bean_results[1].value == 1 && r.bean_results[1].inferred,
        "Z1 should be inferred 1");
  CHECK(crane::count_inferred(r.bean_results) == 1, "inferred count != 1");
}

void test_low_confidence_still_scored() {
  auto tg = make_tg();
  Detections beans = {det(0, 300, 360, 0.90f), det(1, 600, 360, 0.28f),
                      det(2, 900, 360, 0.80f)};
  Detections digits = {det(0, 100), det(1, 300), det(2, 500), det(3, 700),
                       det(4, 900)};
  auto r = tg.generate_frame_with_completion(beans, digits);
  CHECK(r.payload.has_value(), "payload missing: %s", r.reason.c_str());
  CHECK(r.bean_results[2].value == 2 && !r.bean_results[2].inferred,
        "Z2 should be detected");
  CHECK(crane::count_inferred(r.bean_results) == 0, "inferred count != 0");
}

void test_strict_generate_frame() {
  auto tg = make_tg();
  Detections beans = {det(0, 300), det(1, 600), det(2, 900)};
  Detections digits = {det(0, 100), det(1, 300), det(2, 500), det(3, 700),
                       det(4, 900)};
  auto [payload, reason] = tg.generate_frame(beans, digits);
  CHECK(payload.has_value(), "strict payload missing: %s", reason.c_str());
  CHECK((*payload == FramePayload{1, 2, 3, 1, 2, 3, 4, 5}), "wrong payload");
  CHECK(reason == "ok", "reason=%s", reason.c_str());
}

void test_standalone_validate() {
  using ZV = std::map<int, std::optional<std::pair<int, float>>>;
  // All detected
  auto r = crane::validate_and_complete(
      ZV{{1, {{1, 0.9f}}}, {2, {{2, 0.8f}}}, {3, {{3, 0.7f}}}}, {1, 2, 3});
  CHECK(r[1].value == 1 && !r[1].inferred && r[1].conf &&
            std::abs(*r[1].conf - 0.9f) < 1e-6,
        "r[1] wrong");
  CHECK(crane::count_inferred(r) == 0, "inferred != 0");

  // One missing
  r = crane::validate_and_complete(
      ZV{{1, {{1, 0.9f}}}, {2, std::nullopt}, {3, {{3, 0.7f}}}}, {1, 2, 3});
  CHECK(r[2].value == 2 && r[2].inferred && !r[2].conf, "r[2] wrong");
  CHECK(crane::count_inferred(r) == 1, "inferred != 1");

  // Duplicate resolved
  r = crane::validate_and_complete(
      ZV{{1, {{1, 0.55f}}}, {2, {{1, 0.90f}}}, {3, {{3, 0.80f}}}},
      {1, 2, 3});
  CHECK(r[2].value == 1 && !r[2].inferred, "r[2] should be detected 1");
  CHECK(r[1].value == 2 && r[1].inferred, "r[1] should be inferred 2");

  // 5-digit case: two missing
  r = crane::validate_and_complete(
      ZV{{4, {{1, 0.9f}}},
         {5, {{2, 0.8f}}},
         {6, std::nullopt},
         {7, std::nullopt},
         {8, {{5, 0.7f}}}},
      {1, 2, 3, 4, 5});
  CHECK(r[6].value == 3 && r[6].inferred, "r[6] should be inferred 3");
  CHECK(r[7].value == 4 && r[7].inferred, "r[7] should be inferred 4");
  CHECK(crane::count_inferred(r) == 2, "inferred != 2");
}

void test_zone_boundaries() {
  // Boundary line: detection exactly on a line goes to the right zone.
  auto tg = make_tg();
  CHECK(tg.bean_zone(499.9f) == 1, "bean 499.9");
  CHECK(tg.bean_zone(500.0f) == 2, "bean on left_x → zone 2");
  CHECK(tg.bean_zone(800.0f) == 3, "bean on right_x → zone 3");
  CHECK(tg.digit_zone(199.9f) == 4, "digit 199.9");
  CHECK(tg.digit_zone(200.0f) == 5, "digit on x1 → zone 5");
  CHECK(tg.digit_zone(799.9f) == 7, "digit 799.9");
  CHECK(tg.digit_zone(800.0f) == 8, "digit on x4 → zone 8");
}

}  // namespace

int main() {
  test_all_detected();
  test_one_bean_missing();
  test_one_digit_missing();
  test_bean_duplicate_resolved();
  test_too_many_bean_missing();
  test_too_many_digit_missing();
  test_unknown_bean_class_ignored();
  test_low_confidence_still_scored();
  test_strict_generate_frame();
  test_standalone_validate();
  test_zone_boundaries();
  std::printf("%s: %d passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL",
              g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
