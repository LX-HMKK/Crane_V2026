#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace crane {

class VisionConfig;

// Camera interface shared by real V4L2 cameras and synthetic test cameras.
// Mirrors models/camera.py.
class ICamera {
 public:
  virtual ~ICamera() = default;

  // Capture one frame (always 3-channel BGR). Returns false on failure.
  virtual bool capture(cv::Mat& out) = 0;

  virtual bool set_auto_exposure(bool enabled) = 0;
  virtual bool set_exposure(double value) = 0;
  virtual void release() {}

  int width = 0;
  int height = 0;
  int fps = 0;
  double target_fps = 0.0;  // requested fps (0 = unspecified)
  bool auto_exposure = true;
  double exposure = 0.0;
};

using ICameraPtr = std::unique_ptr<ICamera>;

// Real USB camera via cv::VideoCapture (V4L2), with auto-reconnect.
class Camera : public ICamera {
 public:
  Camera(const std::string& device, int width, int height, int buffer_size = 1,
         double fps = 0.0, bool auto_exposure = true, double exposure = -1.0,
         bool grayscale = false, bool reconnect_enabled = true,
         int reconnect_after = 3, double reconnect_interval = 1.0);
  ~Camera() override { release(); }

  bool capture(cv::Mat& out) override;
  bool set_auto_exposure(bool enabled) override;
  bool set_exposure(double value) override;
  void release() override;

  const std::string& device() const { return device_; }

 private:
  void configure_cap();
  bool set_v4l2_control(const std::string& name, int value);
  void maybe_reopen();
  bool reopen();

  std::string device_;
  int req_width_, req_height_;
  bool auto_exposure_requested_ = true;
  double exposure_requested_ = -1.0;  // <0 = unset
  int buffer_size_ = 1;
  bool grayscale_ = false;

  bool reconnect_enabled_ = true;
  int reconnect_after_ = 3;
  double reconnect_interval_ = 1.0;
  int consecutive_failures_ = 0;
  double last_reopen_at_ = 0.0;  // monotonic seconds

  cv::VideoCapture cap_;
  bool opened_ = false;
  std::recursive_mutex io_mutex_;
};

// Synthetic camera for UI testing without real USB cameras.
class MockCamera : public ICamera {
 public:
  MockCamera(int width = 640, int height = 480,
             const std::string& pattern = "color_bars");
  bool capture(cv::Mat& out) override;
  bool set_auto_exposure(bool enabled) override {
    auto_exposure = enabled;
    return true;
  }
  bool set_exposure(double value) override {
    exposure = value;
    return true;
  }

 private:
  std::string pattern_;
  int counter_ = 0;
};

// Camera that repeatedly returns a fixed image file (resized on first read).
class StaticImageCamera : public ICamera {
 public:
  StaticImageCamera(const std::string& image_path, int width, int height);
  bool capture(cv::Mat& out) override;
  bool set_auto_exposure(bool enabled) override {
    auto_exposure = enabled;
    return true;
  }
  bool set_exposure(double value) override {
    exposure = value;
    return true;
  }
  void release() override { frame_.release(); }

 private:
  std::string image_path_;
  cv::Mat frame_;
};

// Factory mirroring runtime.py open_camera(): mock / real / fallback-empty.
ICameraPtr open_camera(const VisionConfig& cfg, const std::string& name,
                       bool mock = false, bool fallback_empty = false);

void list_cameras();

}  // namespace crane
