#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "inference/yolo11.hpp"

namespace crane {

// Model metadata: class count, names, input size, head structure.
// Loaded from ONNX custom metadata (fused path) or the BPU .cfg yaml.
struct ModelMeta {
  int imgsz_h = 640;
  int imgsz_w = 640;
  int nc = 0;
  std::map<int, std::string> names;
  int reg_max = 16;
  std::vector<int> strides = {8, 16, 32};
};

// Full YOLO11 pipeline: letterbox → backend → postprocess → scale back.
// Shared by the ONNX Runtime backend and the RDK BPU backend, mirroring
// models/yolo11_model.py.
class Yolo11Model {
 public:
  Yolo11Model(std::unique_ptr<InferenceBackend> backend, ModelMeta meta,
              float conf_thres, float iou_thres, int pad_value = 127);

  // Run inference on a BGR image; detections in original-image coordinates.
  Detections predict(const cv::Mat& bgr);

  const std::map<int, std::string>& names() const { return meta_.names; }
  const ModelMeta& meta() const { return meta_; }

  // Runtime-tunable thresholds (settings UI), thread-safe.
  void set_conf_thres(float v);
  float conf_thres() const;
  void set_iou_thres(float v);
  float iou_thres() const;

  const std::string& backend_name() const { return backend_name_; }

 private:
  std::unique_ptr<InferenceBackend> backend_;
  ModelMeta meta_;
  int pad_value_;
  std::string backend_name_;

  // kHeads-path caches
  cv::Mat anchors_;
  cv::Mat dfl_proj_;

  mutable std::mutex mutex_;
  float conf_thres_;
  float iou_thres_;
};

// Parse Ultralytics ONNX metadata "names" string, Python-dict style:
//   {0: 'white', 1: 'green', 2: 'yellow'}
std::map<int, std::string> parse_ultralytics_names(const std::string& raw);

}  // namespace crane
