#include "camera.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "common/logging.hpp"
#include "common/config.hpp"

namespace crane {

namespace {

double monotonic_sec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

// ── Camera ──────────────────────────────────────────────────────────────

Camera::Camera(const std::string& device, int width, int height,
               int buffer_size, double fps, bool auto_exposure,
               double exposure, bool grayscale, bool reconnect_enabled,
               int reconnect_after, double reconnect_interval)
    : device_(device),
      req_width_(width),
      req_height_(height),
      auto_exposure_requested_(auto_exposure),
      exposure_requested_(exposure),
      buffer_size_(buffer_size),
      grayscale_(grayscale),
      reconnect_enabled_(reconnect_enabled),
      reconnect_after_(std::max(1, reconnect_after)),
      reconnect_interval_(std::max(0.05, reconnect_interval)) {
  target_fps = fps;
  cap_.open(device_, cv::CAP_V4L2);
  if (!cap_.isOpened()) {
    throw std::runtime_error("Cannot open camera device " + device_);
  }
  opened_ = true;
  configure_cap();
}

void Camera::configure_cap() {
  cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, req_width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, req_height_);
  if (target_fps > 0) {
    cap_.set(cv::CAP_PROP_FPS, target_fps);
  }
  // Some UVC drivers ignore CAP_PROP_FPS and report a different active rate.
  cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);
  width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
  height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
  fps = static_cast<int>(cap_.get(cv::CAP_PROP_FPS));
  set_auto_exposure(auto_exposure_requested_);
  if (exposure_requested_ >= 0) {
    exposure = exposure_requested_;
    if (!auto_exposure_requested_) {
      set_exposure(exposure_requested_);
    }
  }
}

bool Camera::set_auto_exposure(bool enabled) {
  std::lock_guard<std::recursive_mutex> lock(io_mutex_);
  auto_exposure_requested_ = enabled;
  if (!opened_) {
    LOG_WARN("Cannot set auto_exposure=%d, camera %s not open", enabled,
             device_.c_str());
    return false;
  }
  // These RDK USB cameras expose V4L2 menu values 1=manual, 3=auto.
  // Prefer v4l2-ctl: OpenCV's 0.25/0.75 mapping is not applied consistently
  // by their UVC driver.
  int v4l2_value = enabled ? 3 : 1;
  if (set_v4l2_control("auto_exposure", v4l2_value)) {
    auto_exposure = enabled;
    return true;
  }
  double fallback = enabled ? 0.75 : 0.25;
  if (!cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, fallback)) {
    LOG_WARN("Failed to set auto_exposure=%d for camera %s", enabled,
             device_.c_str());
    return false;
  }
  auto_exposure = enabled;
  return true;
}

bool Camera::set_exposure(double value) {
  std::lock_guard<std::recursive_mutex> lock(io_mutex_);
  exposure_requested_ = value;
  if (!opened_) {
    LOG_WARN("Cannot set exposure=%f, camera %s not open", value,
             device_.c_str());
    return false;
  }
  if (set_v4l2_control("exposure_time_absolute",
                       static_cast<int>(std::lround(value)))) {
    exposure = value;
    return true;
  }
  if (!cap_.set(cv::CAP_PROP_EXPOSURE, value)) {
    LOG_WARN("Failed to set exposure=%f for camera %s", value,
             device_.c_str());
    return false;
  }
  exposure = value;
  return true;
}

bool Camera::set_v4l2_control(const std::string& name, int value) {
  if (device_.empty() || device_.rfind("/dev/", 0) != 0) return false;
  std::string cmd = "command -v v4l2-ctl >/dev/null 2>&1 && v4l2-ctl -d '" +
                    device_ + "' --set-ctrl " + name + "=" +
                    std::to_string(value) + " >/dev/null 2>&1";
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    LOG_WARN("v4l2-ctl failed for %s=%d on %s", name.c_str(), value,
             device_.c_str());
    return false;
  }
  return true;
}

bool Camera::capture(cv::Mat& out) {
  bool ok = false;
  {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (!opened_) {
      ++consecutive_failures_;
      maybe_reopen();
    } else {
      try {
        ok = cap_.read(out) && !out.empty();
      } catch (const cv::Exception&) {
        ok = false;
      }
      if (ok) {
        consecutive_failures_ = 0;
      } else {
        ++consecutive_failures_;
        maybe_reopen();
      }
    }
  }
  if (!ok) return false;

  // Some USB cameras deliver single-channel frames; normalize to 3-channel.
  if (out.channels() == 1) {
    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
  }
  // Optionally force grayscale output (still 3 channels) — used for the
  // digit camera whose training data was grayscale.
  if (grayscale_ && out.channels() == 3) {
    cv::Mat gray;
    cv::cvtColor(out, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
  }
  return true;
}

void Camera::maybe_reopen() {
  if (!reconnect_enabled_) return;
  if (consecutive_failures_ < reconnect_after_) return;
  if (monotonic_sec() - last_reopen_at_ < reconnect_interval_) return;
  last_reopen_at_ = monotonic_sec();
  reopen();
}

bool Camera::reopen() {
  if (opened_) {
    try {
      cap_.release();
    } catch (...) {
    }
    opened_ = false;
  }
  try {
    cv::VideoCapture new_cap(device_, cv::CAP_V4L2);
    if (!new_cap.isOpened()) {
      LOG_WARN("Camera %s reopen failed", device_.c_str());
      return false;
    }
    cap_ = std::move(new_cap);
    opened_ = true;
    configure_cap();
    consecutive_failures_ = 0;
    LOG_INFO("Camera %s reconnected", device_.c_str());
    return true;
  } catch (const std::exception& e) {
    LOG_WARN("Camera %s reopen failed: %s", device_.c_str(), e.what());
    if (opened_) {
      try {
        cap_.release();
      } catch (...) {
      }
      opened_ = false;
    }
    return false;
  }
}

void Camera::release() {
  std::lock_guard<std::recursive_mutex> lock(io_mutex_);
  if (!opened_) return;
  cap_.release();
  opened_ = false;
}

// ── MockCamera ────────────────────────────────────────────────────────────

MockCamera::MockCamera(int width, int height, const std::string& pattern)
    : pattern_(pattern) {
  this->width = width;
  this->height = height;
  this->fps = 30;
  auto_exposure = true;
}

bool MockCamera::capture(cv::Mat& out) {
  out = cv::Mat::zeros(height, width, CV_8UC3);

  if (pattern_ == "color_bars") {
    int bar_width = std::max(1, width / 8);
    static const cv::Scalar colors[8] = {
        {0, 0, 255},     {0, 165, 255}, {0, 255, 255}, {0, 255, 0},
        {255, 0, 0},     {255, 0, 255}, {255, 255, 0}, {255, 255, 255},
    };
    for (int i = 0; i < 8; ++i) {
      int x1 = i * bar_width;
      int x2 = std::min(width, x1 + bar_width);
      out.colRange(x1, x2).setTo(colors[i]);
    }
  } else if (pattern_ == "empty") {
    // black frame
  } else {
    int square = std::max(20, std::min(width, height) / 8);
    for (int y = 0; y < height; y += square) {
      for (int x = 0; x < width; x += square) {
        bool white = ((x / square) + (y / square)) % 2 == 0;
        int x2 = std::min(width, x + square);
        int y2 = std::min(height, y + square);
        cv::rectangle(out, {x, y}, {x2 - 1, y2 - 1},
                      white ? cv::Scalar(255, 255, 255)
                            : cv::Scalar(0, 0, 0),
                      cv::FILLED);
      }
    }
  }

  if (pattern_ != "empty") {
    std::string label = "MOCK CAMERA #" + std::to_string(counter_);
    cv::putText(out, label, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                {255, 255, 255}, 2);
  }
  ++counter_;
  return true;
}

// ── StaticImageCamera ─────────────────────────────────────────────────────

StaticImageCamera::StaticImageCamera(const std::string& image_path, int width,
                                     int height)
    : image_path_(image_path) {
  this->width = width;
  this->height = height;
  this->fps = 30;
}

bool StaticImageCamera::capture(cv::Mat& out) {
  if (frame_.empty()) {
    cv::Mat img = cv::imread(image_path_, cv::IMREAD_COLOR);
    if (img.empty()) {
      throw std::runtime_error("Cannot load image " + image_path_);
    }
    if (img.cols != width || img.rows != height) {
      cv::resize(img, img, {width, height});
    }
    frame_ = img;
  }
  frame_.copyTo(out);
  return true;
}

// ── factory / helpers ───────────────────────────────────────────────────────

ICameraPtr open_camera(const VisionConfig& cfg, const std::string& name,
                       bool mock, bool fallback_empty) {
  auto cam_path = [&name](const char* key) {
    return std::vector<std::string>{"cameras", name, key};
  };
  auto rec_path = [&name](const char* key) {
    return std::vector<std::string>{"cameras", name, "reconnect", key};
  };
  int width = cfg.get<int>(cam_path("width"), 640);
  int height = cfg.get<int>(cam_path("height"), 480);
  if (mock) return std::make_unique<MockCamera>(width, height);

  std::string device = cfg.get<std::string>(cam_path("device"), "");
  int buffer_size = cfg.get<int>(cam_path("buffer_size"), 1);
  double fps = cfg.get<double>(cam_path("fps"), 0.0);
  bool auto_exposure = cfg.get<bool>(cam_path("auto_exposure"), true);
  double exposure = cfg.get<double>(cam_path("exposure"), -1.0);
  bool grayscale = cfg.get<bool>(cam_path("grayscale"), false);
  bool reconnect_enabled = cfg.get<bool>(rec_path("enabled"), true);
  int reconnect_after = cfg.get<int>(rec_path("after_failures"), 3);
  double reconnect_interval = cfg.get<double>(rec_path("interval_sec"), 1.0);

  try {
    return std::make_unique<Camera>(
        device, width, height, buffer_size, fps, auto_exposure, exposure,
        grayscale, reconnect_enabled, reconnect_after, reconnect_interval);
  } catch (const std::runtime_error&) {
    if (fallback_empty) {
      LOG_WARN("Cannot open camera device %s; falling back to empty mock "
               "frame",
               device.c_str());
      return std::make_unique<MockCamera>(width, height, "empty");
    }
    throw;
  }
}

void list_cameras() {
  std::printf("Available cameras:\n");
  for (int i = 0; i < 10; ++i) {
    cv::VideoCapture cap(i, cv::CAP_V4L2);
    if (cap.isOpened()) {
      int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
      int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
      std::printf("  /dev/video%d: %dx%d\n", i, w, h);
      cap.release();
    }
  }
}

}  // namespace crane
