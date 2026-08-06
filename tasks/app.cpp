#include "app.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/logging.hpp"
#include "inference/bpu_backend.hpp"
#include "inference/onnx_backend.hpp"
#include "logic/mapping.hpp"
#include "web/web_ui.hpp"

namespace crane {

namespace {

double monotonic_sec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

constexpr int kCaptureOfflineAfterFailures = 5;
constexpr double kCaptureFailureBackoffSec = 0.05;
constexpr int kTaskFailureClearThreshold = 10;
constexpr double kResultUpdateLogIntervalSec = 1.0;
constexpr double kResultFailureLogIntervalSec = 5.0;
constexpr double kInferenceRateWindowSec = 5.0;

LinesConfig lines_from_cfg(const VisionConfig& cfg) {
  // 0.0/0.0 (NaN) is a C2124 constant-expression error under MSVC; use the
  // standard quiet_NaN() so the same value is portable.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  LinesConfig l;
  l.bean_left_x = cfg.get<double>({"lines", "bean", "left_x"}, nan);
  l.bean_right_x = cfg.get<double>({"lines", "bean", "right_x"}, nan);
  l.digit_x1 = cfg.get<double>({"lines", "digit", "x1"}, nan);
  l.digit_x2 = cfg.get<double>({"lines", "digit", "x2"}, nan);
  l.digit_x3 = cfg.get<double>({"lines", "digit", "x3"}, nan);
  l.digit_x4 = cfg.get<double>({"lines", "digit", "x4"}, nan);
  return l;
}

}  // namespace

CompetitionApp::CompetitionApp(VisionConfig cfg, AppOptions opts)
    : cfg_(std::move(cfg)), opts_(std::move(opts)), started_at_(monotonic_sec()) {
  task_update_interval_ =
      cfg_.get<double>({"protocol", "send_interval_sec"}, 0.05);
  rate_window_start_ = monotonic_sec();

  // Push config changes to live devices/models (ConfigBinding on_change).
  for (size_t idx = 0; idx < kNumChannels; ++idx) {
    Channel ch = static_cast<Channel>(idx);
    std::string name = channel_name(ch);
    cfg_.on_change<bool>({"cameras", name, "auto_exposure"},
                         [this, idx](const bool& v) {
                           if (cams_[idx]) cams_[idx]->set_auto_exposure(v);
                         });
    cfg_.on_change<double>({"cameras", name, "exposure"},
                           [this, idx, name](const double& v) {
                             bool auto_exp = cfg_.get<bool>(
                                 {"cameras", name, "auto_exposure"}, true);
                             if (!auto_exp && cams_[idx]) {
                               cams_[idx]->set_exposure(v);
                             }
                           });
    cfg_.on_change<double>({"models", name, "conf_thres"},
                           [this, idx](const double& v) {
                             if (models_[idx]) {
                               models_[idx]->set_conf_thres(
                                   std::round(v * 100.0) / 100.0f);
                             }
                           });
  }
}

CompetitionApp::~CompetitionApp() { stop(); }

void CompetitionApp::open_devices() {
  // Cameras
  for (size_t idx = 0; idx < kNumChannels; ++idx) {
    Channel ch = static_cast<Channel>(idx);
    const std::string name = channel_name(ch);
    const std::string& image =
        ch == Channel::kBean ? opts_.bean_image : opts_.digit_image;
    if (!image.empty()) {
      int w = cfg_.get<int>({"cameras", name, "width"}, 640);
      int h = cfg_.get<int>({"cameras", name, "height"}, 480);
      cams_[idx] = std::make_unique<StaticImageCamera>(image, w, h);
    } else {
      cams_[idx] = open_camera(cfg_, name, opts_.mock_cameras,
                               /*fallback_empty=*/true);
    }
  }
  LOG_INFO("Startup: cameras opened at %.2fs", monotonic_sec() - started_at_);

  // Serial
  if (opts_.use_serial) {
    std::string port = cfg_.get<std::string>({"serial", "port"}, "/dev/ttyS1");
    int baudrate = cfg_.get<int>({"serial", "baudrate"}, 921600);
    double timeout = cfg_.get<double>({"serial", "timeout"}, 0.05);
    double start_timeout =
        cfg_.get<double>({"protocol", "start_timeout_sec"}, 0.3);
    double send_interval =
        cfg_.get<double>({"protocol", "send_interval_sec"}, 0.1);
    bool rec_enabled =
        cfg_.get<bool>({"serial", "reconnect", "enabled"}, true);
    int rec_after =
        cfg_.get<int>({"serial", "reconnect", "after_failures"}, 5);
    double rec_interval =
        cfg_.get<double>({"serial", "reconnect", "interval_sec"}, 1.0);
    try {
      serial_ = SerialThread::for_device(port, baudrate, timeout,
                                         start_timeout, send_interval,
                                         rec_enabled, rec_after, rec_interval);
      LOG_INFO("Serial port %s @ %d 8N1 opened", port.c_str(), baudrate);
    } catch (const std::exception& e) {
      LOG_ERROR("Failed to open serial port %s: %s (running without serial)",
                    port.c_str(), e.what());
      serial_.reset();
    }
  }
}

void CompetitionApp::load_models() {
  std::string backend = opts_.backend;
  if (backend == "auto") {
#ifdef CRANE_BPU_REAL
    backend = "bpu";
#else
    backend = "onnx";
#endif
  }
  int pad = cfg_.get<int>({"inference", "pad"}, 127);
  float glob_conf = cfg_.get<double>({"inference", "conf_thres"}, 0.25);
  float glob_iou = cfg_.get<double>({"inference", "iou_thres"}, 0.45);

  for (size_t idx = 0; idx < kNumChannels; ++idx) {
    const std::string name = channel_name(static_cast<Channel>(idx));
    float conf = cfg_.get<double>({"models", name, "conf_thres"}, glob_conf);
    float iou = cfg_.get<double>({"models", name, "iou_thres"}, glob_iou);
    if (backend == "bpu") {
      std::string bin = cfg_.get<std::string>({"models", name, "bin"}, "");
      std::string bcfg = cfg_.get<std::string>({"models", name, "cfg"}, "");
      models_[idx] = load_bpu_model(bin, bcfg, conf, iou, pad);
    } else {
      std::string onnx = cfg_.get<std::string>({"models", name, "onnx"}, "");
      if (onnx.empty()) {
        // Derive from the .pt path: assets/beans_v6.pt → assets/beans_v6.onnx
        std::string pt = cfg_.get<std::string>({"models", name, "pt"}, "");
        onnx = pt;
        auto pos = onnx.rfind(".pt");
        if (pos != std::string::npos) onnx.replace(pos, 3, ".onnx");
      }
      models_[idx] =
          load_onnx_model(onnx, conf, iou, pad, opts_.ort_threads);
    }
  }
  LOG_INFO("Startup: models loaded (%s backend) at %.2fs", backend.c_str(),
           monotonic_sec() - started_at_);
}

void CompetitionApp::create_task_generator() {
  if (!models_[0] || !models_[1]) {
    throw std::runtime_error(
        "Task generator unavailable; competition mode requires both models");
  }
  auto mapping = cfg_.get_int_map({"mapping", "bean_to_digit"});
  if (mapping.empty()) mapping = kDefaultBeanToDigit;
  task_gen_ = std::make_unique<TaskGenerator>(
      mapping, lines_from_cfg(cfg_), models_[0]->names(), models_[1]->names());
}

void CompetitionApp::warmup() {
  if (opts_.warmup_rounds < 1) return;
  LOG_INFO("Warming up models...");
  for (int r = 0; r < opts_.warmup_rounds; ++r) {
    for (size_t idx = 0; idx < kNumChannels; ++idx) {
      cv::Mat frame;
      if (cams_[idx]->capture(frame) && models_[idx]) {
        models_[idx]->predict(frame);
      }
    }
  }
  LOG_INFO("Startup: models warmed at %.2fs", monotonic_sec() - started_at_);
}

// ── capture worker ─────────────────────────────────────────────────────

void CompetitionApp::capture_worker(Channel ch) {
  const size_t idx = static_cast<size_t>(ch);
  ICamera* cam = cams_[idx].get();
  double fps = cam->target_fps > 0 ? cam->target_fps
                                   : (cam->fps > 0 ? cam->fps : 30.0);
  const double interval = 1.0 / fps;
  double next_capture_at = monotonic_sec();

  while (running_) {
    cv::Mat frame;
    bool ok = false;
    try {
      ok = cam->capture(frame) && !frame.empty();
    } catch (const std::exception& e) {
      LOG_WARN("Camera %s capture exception: %s", channel_name(ch), e.what());
      ok = false;
    }

    if (!ok) {
      bool invalidate = false;
      {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        capture_ready_[idx] = false;
        ++capture_failures_[idx];
        if (!capture_failure_logged_[idx]) {
          LOG_WARN("Camera %s capture failure (%d consecutive)",
                   channel_name(ch), capture_failures_[idx]);
          capture_failure_logged_[idx] = true;
        }
        invalidate = capture_failures_[idx] >= kCaptureOfflineAfterFailures;
      }
      if (invalidate) invalidate_camera_detections(ch);
      std::this_thread::sleep_for(
          std::chrono::duration<double>(kCaptureFailureBackoffSec));
      next_capture_at = monotonic_sec() + interval;
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(capture_mutex_);
      captured_frames_[idx] = frame;
      capture_ready_[idx] = true;
      capture_failures_[idx] = 0;
      capture_failure_logged_[idx] = false;
      ++capture_sequences_[idx];
    }
    submit_frame_for_inference(ch, frame);

    next_capture_at += interval;
    double delay = next_capture_at - monotonic_sec();
    if (delay > 0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(delay));
    } else {
      next_capture_at = monotonic_sec();  // fell behind; don't accumulate
    }
  }
}

void CompetitionApp::submit_frame_for_inference(Channel ch,
                                                const cv::Mat& frame) {
  {
    std::lock_guard<std::mutex> lock(infer_mutex_);
    infer_pending_[static_cast<size_t>(ch)] = frame;  // latest-slot
  }
  infer_cv_.notify_all();
}

bool CompetitionApp::camera_capture_healthy(Channel ch) const {
  std::lock_guard<std::mutex> lock(capture_mutex_);
  return capture_failures_[static_cast<size_t>(ch)] <
         kCaptureOfflineAfterFailures;
}

void CompetitionApp::invalidate_camera_detections(Channel ch) {
  std::lock_guard<std::mutex> lock(result_mutex_);
  latest_dets_[static_cast<size_t>(ch)].clear();
}

// ── inference worker ───────────────────────────────────────────────────

void CompetitionApp::inference_worker(Channel ch) {
  const size_t idx = static_cast<size_t>(ch);
  Yolo11Model* model = models_[idx].get();
  while (running_) {
    cv::Mat frame;
    {
      std::unique_lock<std::mutex> lock(infer_mutex_);
      // Consume-side gate: frames are only fed into the model while the MCU
      // keeps START alive (or DEBUG forces recognition). The capture threads
      // keep refreshing the latest slot, so on START-on the worker picks up
      // the freshest frame. The timed wait is a backstop: it re-checks the
      // gate even when no capture notify arrives (e.g. START arriving while
      // cameras are in failure backoff), decoupling START latency from the
      // capture rate. MUST honour the return value: wait_for's predicate
      // result, not the mere fact that it returned. On timeout with the gate
      // closed the predicate is false and the pending frame must NOT be
      // consumed (otherwise the gate is a no-op and standby still infers).
      bool ready = infer_cv_.wait_for(lock, std::chrono::milliseconds(50), [&] {
        return !running_ || (!infer_pending_[idx].empty() &&
                             recognition_enabled());
      });
      if (!running_) return;
      if (!ready) continue;
      frame = std::move(infer_pending_[idx]);
      infer_pending_[idx].release();
    }

    Detections dets;
    try {
      dets = model->predict(frame);
    } catch (const std::exception& e) {
      LOG_ERROR("Model %s inference failed: %s", channel_name(ch), e.what());
      continue;
    }
    if (!running_) return;

    // No publish-side gate: a predict that started while START was active may
    // finish just after START dropped. Publishing it is intentional — it is
    // the last valid result and becomes the sticky cached frame (TX is
    // separately START-gated, so nothing transmits during standby).
    if (camera_capture_healthy(ch)) {
      std::lock_guard<std::mutex> lock(result_mutex_);
      latest_dets_[idx] = std::move(dets);
    } else {
      continue;
    }

    on_inference_done(ch);

    // rate meter
    {
      std::lock_guard<std::mutex> lock(rate_mutex_);
      ++rate_counts_[idx];
      double now = monotonic_sec();
      double elapsed = now - rate_window_start_;
      if (elapsed >= kInferenceRateWindowSec) {
        rate_fps_[0] = rate_counts_[0] / elapsed;
        rate_fps_[1] = rate_counts_[1] / elapsed;
        LOG_INFO("Inference throughput: bean=%.2f fps, digit=%.2f fps, "
                 "total=%.2f fps (%.1fs window)",
                 rate_fps_[0], rate_fps_[1], rate_fps_[0] + rate_fps_[1],
                 elapsed);
        rate_counts_ = {0, 0};
        rate_window_start_ = now;
      }
    }
  }
}

void CompetitionApp::on_inference_done(Channel /*ch*/) {
  std::lock_guard<std::mutex> lock(task_mutex_);
  double now = monotonic_sec();
  if (now - last_task_update_at_ < task_update_interval_) return;
  last_task_update_at_ = now;
  update_task_from_latest_detections();
}

void CompetitionApp::update_task_from_latest_detections() {
  // Both cameras must be healthy; otherwise keep the previous result
  // (sticky).
  if (!camera_capture_healthy(Channel::kBean) ||
      !camera_capture_healthy(Channel::kDigit)) {
    return;
  }

  Detections bean_dets, digit_dets;
  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    bean_dets = latest_dets_[0];
    digit_dets = latest_dets_[1];
  }

  // Rebuild the task generator on the fly so line edits take effect
  // immediately (construction is trivial).
  TaskGenerator gen(task_gen_->mapping(), lines_from_cfg(cfg_),
                    models_[0]->names(), models_[1]->names());

  float conf_threshold =
      cfg_.get<double>({"logic", "confidence_threshold"}, 0.5);
  int max_beans = cfg_.get<int>({"logic", "max_inferred_beans"}, 1);
  int max_digits = cfg_.get<int>({"logic", "max_inferred_digits"}, 2);

  auto result = gen.generate_frame_with_completion(
      bean_dets, digit_dets, conf_threshold, max_beans, max_digits);

  double now = monotonic_sec();
  if (result.payload.has_value()) {
    consecutive_task_failures_ = 0;
    last_reason_ = result.reason;
    bool had_frame = last_frame_.has_value();
    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      zone_results_[0] = std::move(result.bean_results);
      zone_results_[1] = std::move(result.digit_results);
    }
    if (!had_frame || *result.payload != *last_frame_) {
      if (serial_) serial_->set_current_frame(*result.payload);
      last_frame_ = *result.payload;
      if (!had_frame) {
        LOG_INFO("Startup: first valid result ready at %.2fs",
                 now - started_at_);
      }
      if (now - last_result_log_at_ >= kResultUpdateLogIntervalSec) {
        const auto& f = *result.payload;
        int inferred = 0;
        for (const auto& kv : zone_results_[0]) inferred += kv.second.inferred;
        for (const auto& kv : zone_results_[1]) inferred += kv.second.inferred;
        LOG_INFO("Result updated: C1-C3=%d %d %d, N4-N8=%d %d %d %d %d "
                 "(%d inferred)",
                 f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], inferred);
        last_result_log_at_ = now;
      }
    }
  } else {
    ++consecutive_task_failures_;
    last_reason_ = result.reason;
    if (consecutive_task_failures_ < kTaskFailureClearThreshold) {
      LOG_WARN("Result generation failed (%d/%d): %s",
               consecutive_task_failures_, kTaskFailureClearThreshold,
               result.reason.c_str());
    } else {
      if (serial_) serial_->set_current_frame(std::nullopt);
      last_frame_.reset();
      if (now - last_failure_log_at_ >= kResultFailureLogIntervalSec) {
        LOG_ERROR("No valid result after %d scans; clearing serial frame: %s",
                  consecutive_task_failures_, result.reason.c_str());
        last_failure_log_at_ = now;
      }
    }
  }
}

// ── run / stop ─────────────────────────────────────────────────────────

int CompetitionApp::run() {
  open_devices();
  load_models();
  create_task_generator();
  warmup();

  if (serial_) serial_->start();
  running_ = true;
  for (size_t idx = 0; idx < kNumChannels; ++idx) {
    Channel ch = static_cast<Channel>(idx);
    capture_threads_[idx] =
        std::thread(&CompetitionApp::capture_worker, this, ch);
    inference_threads_[idx] =
        std::thread(&CompetitionApp::inference_worker, this, ch);
  }
  if (opts_.web_enabled) {
    web_ = std::make_unique<WebUi>(*this, opts_.web_listen, opts_.web_port);
    web_->start();
  }
  LOG_INFO("Startup: ready at %.2fs", monotonic_sec() - started_at_);

  while (running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Shutdown
  for (auto& t : capture_threads_)
    if (t.joinable()) t.join();
  {
    std::lock_guard<std::mutex> lock(infer_mutex_);
    infer_pending_[0].release();
    infer_pending_[1].release();
  }
  infer_cv_.notify_all();
  for (auto& t : inference_threads_)
    if (t.joinable()) t.join();
  if (web_) web_->stop();
  if (serial_) serial_->stop();
  for (auto& cam : cams_)
    if (cam) cam->release();
  LOG_INFO("Shutdown complete");
  return 0;
}

void CompetitionApp::stop() {
  running_ = false;
  infer_cv_.notify_all();
}

bool CompetitionApp::recognition_enabled() const {
  if (debug_mode_.load(std::memory_order_relaxed)) return true;
  // No usable serial backend (--no-serial dev mode, or a port that failed to
  // open at boot — e.g. the Windows dev-verification stub which always
  // throws): there is no MCU to signal START, so recognition stays OFF. This
  // is the board-idle state; it also stops a Windows dev run from pinning both
  // inference workers at 100% CPU during UI calibration. Use the DEBUG toggle
  // to force recognition for calibration/verification.
  if (!serial_ || !serial_->connected()) return false;
  // The serial RX loop is in its 1s error-backoff pause: it cannot read START
  // during that window, so a strict START check would freeze perception. Fail
  // open only while the loop is unhealthy (a live port that is erroring), so a
  // transient UART blip does not kill a running round. TX stays separately
  // START-gated, so nothing transmits while the MCU is not signalling.
  if (!serial_->rx_healthy()) return true;
  return serial_->is_start_active();
}

void CompetitionApp::set_debug_mode(bool on) {
  bool changed = debug_mode_.exchange(on, std::memory_order_relaxed) != on;
  if (changed) {
    LOG_INFO("Debug mode %s (recognition %s)", on ? "enabled" : "disabled",
             on ? "forced ON" : "gated by serial START");
  }
  infer_cv_.notify_all();  // re-evaluate the recognition gate immediately
}

// ── once / bench ───────────────────────────────────────────────────────

int CompetitionApp::run_once() {
  open_devices();
  load_models();
  create_task_generator();

  const int frames = cfg_.get<int>({"scan", "frames"}, 2);
  Detections all_bean, all_digit;
  int failures = 0;
  for (int i = 0; i < frames; ++i) {
    cv::Mat fb, fd;
    bool ok_b = cams_[0]->capture(fb);
    bool ok_d = cams_[1]->capture(fd);
    if (!ok_b || !ok_d) {
      ++failures;
      LOG_WARN("Camera capture failure (bean_ok=%d, digit_ok=%d)", ok_b,
               ok_d);
      continue;
    }
    auto db = models_[0]->predict(fb);
    auto dd = models_[1]->predict(fd);
    all_bean.insert(all_bean.end(), db.begin(), db.end());
    all_digit.insert(all_digit.end(), dd.begin(), dd.end());
  }
  if (failures >= frames) {
    std::fprintf(stderr, "SCAN FAILED: too many camera capture failures\n");
    return 1;
  }

  std::printf("Bean detections (%zu):\n", all_bean.size());
  for (const auto& d : all_bean) {
    std::printf("  [%.1f %.1f %.1f %.1f] conf=%.3f cls=%d (%s)\n", d.x1,
                d.y1, d.x2, d.y2, d.conf, d.cls_id,
                models_[0]->names().count(d.cls_id)
                    ? models_[0]->names().at(d.cls_id).c_str()
                    : "?");
  }
  std::printf("Digit detections (%zu):\n", all_digit.size());
  for (const auto& d : all_digit) {
    std::printf("  [%.1f %.1f %.1f %.1f] conf=%.3f cls=%d (%s)\n", d.x1,
                d.y1, d.x2, d.y2, d.conf, d.cls_id,
                models_[1]->names().count(d.cls_id)
                    ? models_[1]->names().at(d.cls_id).c_str()
                    : "?");
  }

  float conf_threshold =
      cfg_.get<double>({"logic", "confidence_threshold"}, 0.5);
  int max_beans = cfg_.get<int>({"logic", "max_inferred_beans"}, 1);
  int max_digits = cfg_.get<int>({"logic", "max_inferred_digits"}, 2);
  auto result = task_gen_->generate_frame_with_completion(
      all_bean, all_digit, conf_threshold, max_beans, max_digits);

  if (!result.payload.has_value()) {
    std::fprintf(stderr, "SCAN FAILED: %s\n", result.reason.c_str());
    return 1;
  }
  const auto& f = *result.payload;
  std::printf("SCAN RESULT: C1-C3=%d %d %d N4-N8=%d %d %d %d %d\n", f[0],
              f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
  auto frame = SerialPort::encode_frame(f);
  std::printf("FRAME: ");
  for (uint8_t b : frame) std::printf("%02X ", b);
  std::printf("\n");
  return 0;
}

int CompetitionApp::bench(int rounds) {
  opts_.use_serial = false;
  open_devices();
  load_models();

  for (size_t idx = 0; idx < kNumChannels; ++idx) {
    cv::Mat frame;
    if (!cams_[idx]->capture(frame)) {
      LOG_ERROR("bench: camera %s capture failed", channel_name(
                                               static_cast<Channel>(idx)));
      return 1;
    }
    // warmup
    for (int i = 0; i < 3; ++i) models_[idx]->predict(frame);
    std::vector<double> times;
    times.reserve(rounds);
    for (int i = 0; i < rounds; ++i) {
      double t0 = monotonic_sec();
      models_[idx]->predict(frame);
      times.push_back((monotonic_sec() - t0) * 1000.0);
    }
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    std::printf("%s model (%s): avg=%.2fms p50=%.2fms min=%.2fms "
                "max=%.2fms over %d rounds (%.1f fps)\n",
                channel_name(static_cast<Channel>(idx)),
                models_[idx]->backend_name().c_str(), sum / rounds,
                times[rounds / 2], times.front(), times.back(), rounds,
                1000.0 / (sum / rounds));
  }
  return 0;
}

// ── status / rendering for the web UI ──────────────────────────────────

CompetitionApp::Status CompetitionApp::status() const {
  Status s;
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    s.bean_online = capture_failures_[0] < kCaptureOfflineAfterFailures;
    s.digit_online = capture_failures_[1] < kCaptureOfflineAfterFailures;
  }
  s.serial_present = serial_ != nullptr;
  s.serial_connected = serial_ && serial_->connected();
  s.start_active = serial_ && serial_->is_start_active();
  s.debug_mode = debug_mode();
  s.recognition_active = recognition_enabled();
  {
    std::lock_guard<std::mutex> lock(rate_mutex_);
    s.bean_infer_fps = rate_fps_[0];
    s.digit_infer_fps = rate_fps_[1];
  }
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    s.frame = last_frame_;
    s.last_reason = last_reason_;
  }
  return s;
}

bool CompetitionApp::render_view(Channel ch, cv::Mat& out) {
  const size_t idx = static_cast<size_t>(ch);
  const std::string name = channel_name(ch);
  cv::Mat frame;
  bool online;
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    frame = captured_frames_[idx];
    online = capture_failures_[idx] < kCaptureOfflineAfterFailures;
  }

  Detections dets;
  std::map<int, ZoneResult> zones;
  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    dets = latest_dets_[idx];
    zones = zone_results_[idx];
  }
  LinesConfig lines = lines_from_cfg(cfg_);

  if (frame.empty() || !online) {
    frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    std::string text = online ? "WAITING FOR CAMERA..."
                              : (name == "bean" ? "BEAN CAMERA OFFLINE"
                                                : "DIGIT CAMERA OFFLINE");
    cv::Scalar color = online ? cv::Scalar(160, 160, 160)
                              : cv::Scalar(0, 0, 255);
    cv::putText(frame, text, {40, 360}, cv::FONT_HERSHEY_SIMPLEX, 1.2,
                color, 3);
  } else {
    frame = frame.clone();
  }

  // Guide lines
  if (ch == Channel::kBean) {
    for (double x : {lines.bean_left_x, lines.bean_right_x}) {
      if (!std::isnan(x)) {
        cv::line(frame, {static_cast<int>(x), 0},
                 {static_cast<int>(x), frame.rows}, {0, 255, 0}, 2);
      }
    }
  } else {
    for (double x :
         {lines.digit_x1, lines.digit_x2, lines.digit_x3, lines.digit_x4}) {
      if (!std::isnan(x)) {
        cv::line(frame, {static_cast<int>(x), 0},
                 {static_cast<int>(x), frame.rows}, {255, 0, 0}, 2);
      }
    }
  }

  // Detections: yellow = detected, orange = inferred zone.
  TaskGenerator zone_gen(kDefaultBeanToDigit, lines, models_[0]->names(),
                         models_[1]->names());
  for (const auto& d : dets) {
    int zone = ch == Channel::kBean ? zone_gen.bean_zone(d.cx())
                                    : zone_gen.digit_zone(d.cx());
    bool inferred = zones.count(zone) && zones[zone].inferred;
    cv::Scalar color = inferred ? cv::Scalar(0, 140, 255)
                                : cv::Scalar(0, 255, 255);
    cv::rectangle(frame, {static_cast<int>(d.x1), static_cast<int>(d.y1)},
                  {static_cast<int>(d.x2), static_cast<int>(d.y2)}, color,
                  2);
    std::string cls_name =
        models_[idx]->names().count(d.cls_id)
            ? models_[idx]->names().at(d.cls_id)
            : "?";
    for (auto& c : cls_name) c = std::toupper(c);
    char label[64];
    std::snprintf(label, sizeof(label), "%s%s %.2f", inferred ? "~" : "",
                  cls_name.c_str(), d.conf);
    cv::putText(frame, label,
                {static_cast<int>(d.x1),
                 std::max(0, static_cast<int>(d.y1) - 4)},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
  }

  // Inferred-zone virtual labels.
  std::vector<double> boundaries;
  int first_zone;
  if (ch == Channel::kBean) {
    boundaries = {0, lines.bean_left_x, lines.bean_right_x,
                  static_cast<double>(frame.cols)};
    first_zone = 1;
  } else {
    boundaries = {0, lines.digit_x1, lines.digit_x2, lines.digit_x3,
                  lines.digit_x4, static_cast<double>(frame.cols)};
    first_zone = 4;
  }
  for (const auto& kv : zones) {
    if (!kv.second.inferred) continue;
    int zone = kv.first;
    int bidx = zone - first_zone;
    if (bidx < 0 || bidx + 1 >= static_cast<int>(boundaries.size()))
      continue;
    double mid = (boundaries[bidx] + boundaries[bidx + 1]) / 2.0;
    std::string value_name;
    if (ch == Channel::kBean) {
      static const char* kColorNames[] = {"", "YELLOW", "GREEN", "WHITE"};
      if (kv.second.value >= 1 && kv.second.value <= 3)
        value_name = kColorNames[kv.second.value];
    } else {
      value_name = std::to_string(kv.second.value);
    }
    std::string text = "INFER Z" + std::to_string(zone) + ": " + value_name;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2,
                                  &baseline);
    int tx = std::min(
        std::max(4, static_cast<int>(mid) - ts.width / 2),
        std::max(4, frame.cols - ts.width - 4));
    int ty = 24;
    cv::rectangle(frame, {tx - 2, ty - ts.height - 4},
                  {tx + ts.width + 2, ty + baseline + 2}, {0, 0, 0},
                  cv::FILLED);
    cv::putText(frame, text, {tx, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                {0, 140, 255}, 2);
  }

  // Status / fps are rendered by the HTML bottom status bar (index.html),
  // matching the reference Tkinter layout; the video frame carries only the
  // calibration overlay (guide lines, detections, inferred-zone labels).
  out = std::move(frame);
  return true;
}

void CompetitionApp::apply_runtime_overrides() {
  for (size_t idx = 0; idx < kNumChannels; ++idx) {
    std::string name = channel_name(static_cast<Channel>(idx));
    if (cams_[idx]) {
      bool auto_exp =
          cfg_.get<bool>({"cameras", name, "auto_exposure"}, true);
      cams_[idx]->set_auto_exposure(auto_exp);
      if (!auto_exp) {
        double exp = cfg_.get<double>({"cameras", name, "exposure"}, -1.0);
        if (exp >= 0) cams_[idx]->set_exposure(exp);
      }
    }
    if (models_[idx]) {
      double conf =
          cfg_.get<double>({"models", name, "conf_thres"}, -1.0);
      if (conf >= 0) models_[idx]->set_conf_thres(conf);
    }
  }
}

}  // namespace crane
