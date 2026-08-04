#pragma once

#include <memory>
#include <string>

#include "inference/model.hpp"
#include "inference/yolo11.hpp"

namespace crane {

// RDK X5 BPU backend via libDNN (hbDNN C API), for hb_mapper-compiled
// YOLO11 .bin models with 6 raw heads ([s/m/l]_bbox + [s/m/l]_cls).
//
// Built on all platforms: on the RDK X5 (aarch64 with /usr/lib/hbbpu/
// libdnn.so) it links the real library; elsewhere it links a stub so the
// binary still builds and the backend reports itself unavailable.
class BpuBackend : public InferenceBackend {
 public:
  // bin_path: .bin model; cfg_path: deployment yaml (imgsz/nc/names/...).
  BpuBackend(const std::string& bin_path, const std::string& cfg_path);
  ~BpuBackend() override;

  RawOutputs run(const cv::Mat& letterboxed_bgr) override;
  RawOutputs::Layout output_layout() const override {
    return RawOutputs::Layout::kHeads;
  }
  std::string name() const override { return "bpu-libdnn"; }

  ModelMeta meta() const { return meta_; }
  bool available() const { return available_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ModelMeta meta_;
  bool available_ = false;
};

// Factory helper mirroring runtime.py load_models() for the BPU path.
std::unique_ptr<Yolo11Model> load_bpu_model(const std::string& bin_path,
                                            const std::string& cfg_path,
                                            float conf_thres,
                                            float iou_thres, int pad_value);

}  // namespace crane
