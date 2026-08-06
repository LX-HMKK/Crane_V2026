// Golden integration test: ONNX backend detections on static images must
// match the Python onnxruntime reference within tolerance, and the full
// pipeline must produce the expected protocol payload.
//
// Reference values were produced by onnxruntime 1.23.2 (Python) with the
// same letterbox (pad=127), conf_thres=0.25, iou_thres=0.45.
// Tolerances absorb cv::resize INTER_LINEAR differences between OpenCV
// 4.5.4 (WSL2 apt, used here) and 5.0 (pip, used for the reference):
// sub-pixel coordinate shifts and ~0.2% confidence wobble, which cannot
// affect zone assignment (zones are ≥250 px wide).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "common/config.hpp"
#include "inference/onnx_backend.hpp"
#include "logic/mapping.hpp"
#include "io/serial_port.hpp"
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

struct Golden {
  float x1, y1, x2, y2, conf;
  int cls_id;
};

bool close_float(float a, float b, float tol) {
  return std::fabs(a - b) <= tol;
}

void check_dets(const char* name, const crane::Detections& dets,
                const std::vector<Golden>& golden) {
  CHECK(dets.size() == golden.size(), "%s: %zu dets, want %zu", name,
        dets.size(), golden.size());
  size_t n = std::min(dets.size(), golden.size());
  for (size_t i = 0; i < n; ++i) {
    const auto& d = dets[i];
    const auto& g = golden[i];
    CHECK(d.cls_id == g.cls_id, "%s[%zu]: cls %d want %d", name, i,
          d.cls_id, g.cls_id);
    CHECK(close_float(d.x1, g.x1, 1.5f) && close_float(d.y1, g.y1, 1.5f) &&
              close_float(d.x2, g.x2, 1.5f) && close_float(d.y2, g.y2, 1.5f),
          "%s[%zu]: box [%.2f %.2f %.2f %.2f] want [%.2f %.2f %.2f %.2f]",
          name, i, d.x1, d.y1, d.x2, d.y2, g.x1, g.y1, g.x2, g.y2);
    CHECK(close_float(d.conf, g.conf, 5e-3f), "%s[%zu]: conf %.4f want %.4f",
          name, i, d.conf, g.conf);
  }
}

}  // namespace

int main() {
  // ── 1. bean model on beans_2.jpg ─────────────────────────────────────
  auto bean_model = crane::load_onnx_model("assets/beans_v6.onnx",
                                           /*conf_thres=*/0.25f,
                                           /*iou_thres=*/0.45f,
                                           /*pad_value=*/127);
  cv::Mat bean_img = cv::imread("tests/data/beans_2.jpg");
  if (bean_img.empty()) {
    std::printf("FAIL: cannot load tests/data/beans_2.jpg\n");
    return 1;
  }
  auto bean_dets = bean_model->predict(bean_img);
  check_dets("beans", bean_dets,
             {{757.613f, 218.531f, 896.910f, 302.001f, 0.920f, 1},
              {541.395f, 231.574f, 689.181f, 323.440f, 0.919f, 0},
              {309.299f, 210.079f, 451.072f, 293.373f, 0.915f, 2}});

  // ── 2. digit model on num_2.jpg ──────────────────────────────────────
  auto digit_model = crane::load_onnx_model("assets/num_v6.onnx",
                                            /*conf_thres=*/0.25f,
                                            /*iou_thres=*/0.45f,
                                            /*pad_value=*/127);
  cv::Mat digit_img = cv::imread("tests/data/num_2.jpg");
  if (digit_img.empty()) {
    std::printf("FAIL: cannot load tests/data/num_2.jpg\n");
    return 1;
  }
  auto digit_dets = digit_model->predict(digit_img);
  check_dets("num", digit_dets,
             {{894.026f, 443.646f, 920.717f, 477.477f, 0.934f, 2},
              {643.514f, 462.691f, 669.905f, 497.494f, 0.933f, 4},
              {397.962f, 482.262f, 426.468f, 514.387f, 0.925f, 3},
              {1180.316f, 459.968f, 1203.983f, 500.447f, 0.906f, 1},
              {143.587f, 534.246f, 175.970f, 571.826f, 0.788f, 0}});

  // ── 3. full pipeline: payload + encoded frame ────────────────────────
  crane::VisionConfig cfg("configs/vision.yaml");
  crane::LinesConfig lines;
  lines.bean_left_x = cfg.get<double>({"lines", "bean", "left_x"}, 0);
  lines.bean_right_x = cfg.get<double>({"lines", "bean", "right_x"}, 0);
  lines.digit_x1 = cfg.get<double>({"lines", "digit", "x1"}, 0);
  lines.digit_x2 = cfg.get<double>({"lines", "digit", "x2"}, 0);
  lines.digit_x3 = cfg.get<double>({"lines", "digit", "x3"}, 0);
  lines.digit_x4 = cfg.get<double>({"lines", "digit", "x4"}, 0);

  auto mapping = cfg.get_int_map({"mapping", "bean_to_digit"});
  if (mapping.empty()) mapping = crane::kDefaultBeanToDigit;
  crane::TaskGenerator tg(mapping, lines, bean_model->names(),
                          digit_model->names());
  auto result = tg.generate_frame_with_completion(bean_dets, digit_dets);
  CHECK(result.payload.has_value(), "payload missing: %s",
        result.reason.c_str());
  if (result.payload.has_value()) {
    crane::FramePayload want{1, 3, 2, 1, 4, 5, 3, 2};
    CHECK(*result.payload == want,
          "payload %d %d %d %d %d %d %d %d, want 1 3 2 1 4 5 3 2",
          (*result.payload)[0], (*result.payload)[1], (*result.payload)[2],
          (*result.payload)[3], (*result.payload)[4], (*result.payload)[5],
          (*result.payload)[6], (*result.payload)[7]);
    auto frame = crane::SerialPort::encode_frame(*result.payload);
    std::array<uint8_t, 12> want_frame = {0xA5, 0x5A, 0x01, 0x03, 0x02, 0x01,
                                          0x04, 0x05, 0x03, 0x02, 0x0D,
                                          0x0A};
    CHECK(frame == want_frame, "encoded frame mismatch");
  }

  std::printf("%s: %d passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL",
              g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
