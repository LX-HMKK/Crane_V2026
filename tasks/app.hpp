#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include "io/camera.hpp"
#include "common/types.hpp"
#include "common/config.hpp"
#include "inference/model.hpp"
#include "io/serial_port.hpp"
#include "logic/task_generator.hpp"

namespace crane {

class WebUi;

struct AppOptions {
  bool mock_cameras = false;
  std::string bean_image;   // static image overrides
  std::string digit_image;
  bool use_serial = true;
  bool web_enabled = true;
  std::string web_listen = "0.0.0.0";
  int web_port = 8080;
  std::string backend = "auto";  // auto | onnx | bpu
  int ort_threads = 0;           // 0 = ONNX Runtime default
  int warmup_rounds = 1;
};

// Camera/model channel identifier.
enum class Channel : size_t { kBean = 0, kDigit = 1 };
inline constexpr size_t kNumChannels = 2;
inline const char* channel_name(Channel c) {
  return c == Channel::kBean ? "bean" : "digit";
}
inline Channel other_channel(Channel c) {
  return c == Channel::kBean ? Channel::kDigit : Channel::kBean;
}

// Competition orchestration: capture threads + inference threads + task
// combiner + serial output + optional web UI. Mirrors CompetitionUI's
// thread model and the headless competition loop of main_arm64.py.
class CompetitionApp {
 public:
  CompetitionApp(VisionConfig cfg, AppOptions opts);
  ~CompetitionApp();

  // Blocking competition loop; returns when stop() is called (signal or
  // web EXIT). Returns process exit code.
  int run();
  void stop();

  // Single scan on static/mock cameras: print detections + payload.
  // Returns 0 on a valid payload, 1 otherwise.
  int run_once();

  // Inference benchmark on static/mock cameras.
  int bench(int rounds);

  // ── shared state access for the web UI ─────────────────────────────
  struct Status {
    bool bean_online = false;
    bool digit_online = false;
    bool serial_present = false;
    bool serial_connected = false;
    bool start_active = false;
    double bean_infer_fps = 0;
    double digit_infer_fps = 0;
    std::optional<FramePayload> frame;
    std::string last_reason;
  };
  Status status() const;

  // Latest annotated view (BGR). Returns false when nothing captured yet.
  bool render_view(Channel ch, cv::Mat& out);

  VisionConfig& cfg() { return cfg_; }
  bool save_cfg() { return cfg_.save(); }
  void apply_runtime_overrides();  // push cfg → cameras/models
  void request_exit() { stop(); }

 private:
  void open_devices();
  void load_models();
  void create_task_generator();
  void warmup();

  void capture_worker(Channel ch);
  void inference_worker(Channel ch);
  void submit_frame_for_inference(Channel ch, const cv::Mat& frame);
  bool camera_capture_healthy(Channel ch) const;
  void invalidate_camera_detections(Channel ch);
  void on_inference_done(Channel ch);
  void update_task_from_latest_detections();

  VisionConfig cfg_;
  AppOptions opts_;

  // Shutdown-signaling primitives are declared BEFORE web_/serial_ so they
  // are destroyed AFTER them (reverse declaration order): a web exit-thread
  // joined from ~WebUi may still call stop() while WebUi is being destroyed.
  std::atomic<bool> running_{false};
  std::mutex infer_mutex_;
  std::condition_variable infer_cv_;

  ICameraPtr cams_[kNumChannels];
  std::unique_ptr<Yolo11Model> models_[kNumChannels];
  std::unique_ptr<TaskGenerator> task_gen_;
  std::unique_ptr<SerialThread> serial_;
  std::unique_ptr<WebUi> web_;

  std::array<std::thread, kNumChannels> capture_threads_;
  std::array<std::thread, kNumChannels> inference_threads_;

  // ── capture slots (latest-frame semantics) ─────────────────────────
  mutable std::mutex capture_mutex_;
  std::array<cv::Mat, kNumChannels> captured_frames_;
  std::array<bool, kNumChannels> capture_ready_ = {false, false};
  std::array<int, kNumChannels> capture_failures_ = {0, 0};
  std::array<uint64_t, kNumChannels> capture_sequences_ = {0, 0};
  std::array<bool, kNumChannels> capture_failure_logged_ = {false, false};

  // ── inference pending slots ────────────────────────────────────────
  std::array<cv::Mat, kNumChannels> infer_pending_;

  // ── published detections ───────────────────────────────────────────
  mutable std::mutex result_mutex_;
  std::array<Detections, kNumChannels> latest_dets_;
  std::array<std::map<int, ZoneResult>, kNumChannels> zone_results_;

  // ── task combiner ──────────────────────────────────────────────────
  mutable std::mutex task_mutex_;
  double last_task_update_at_ = 0.0;
  double task_update_interval_ = 0.05;
  std::optional<FramePayload> last_frame_;
  int consecutive_task_failures_ = 0;
  std::string last_reason_;
  double last_result_log_at_ = 0.0;
  double last_failure_log_at_ = 0.0;

  // ── inference rate meter ───────────────────────────────────────────
  mutable std::mutex rate_mutex_;
  std::array<int, kNumChannels> rate_counts_ = {0, 0};
  double rate_window_start_ = 0.0;
  std::array<double, kNumChannels> rate_fps_ = {0, 0};

  double started_at_ = 0.0;
};

}  // namespace crane
