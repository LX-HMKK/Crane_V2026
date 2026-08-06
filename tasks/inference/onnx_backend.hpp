#pragma once

#include <memory>
#include <string>

#include "inference/model.hpp"
#include "inference/yolo11.hpp"

namespace crane {

// ONNX Runtime backend for fused Ultralytics exports ([1, 4+nc, N] output,
// DFL decoded + sigmoid in-graph). Default CPU execution provider; pass
// exec_provider="cuda" to register the CUDA EP (Windows dev-verification
// builds on an NVIDIA GPU). The CUDA path fails loudly — never falls back to
// CPU — when the CUDA runtime/cuDNN are missing or mismatched.
class OnnxBackend : public InferenceBackend {
 public:
  explicit OnnxBackend(const std::string& onnx_path, int intra_threads = 0,
                       const std::string& exec_provider = "cpu");
  ~OnnxBackend() override;

  RawOutputs run(const cv::Mat& letterboxed_bgr) override;
  RawOutputs::Layout output_layout() const override {
    return RawOutputs::Layout::kFused;
  }
  std::string name() const override { return "onnxruntime"; }

  // Ultralytics metadata read from the ONNX file.
  ModelMeta meta() const { return meta_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ModelMeta meta_;
};

// Factory helper: load backend + build ModelMeta from ONNX metadata.
// exec_provider: "cpu" (default, deterministic for golden) or "cuda".
std::unique_ptr<Yolo11Model> load_onnx_model(const std::string& onnx_path,
                                             float conf_thres,
                                             float iou_thres, int pad_value,
                                             int intra_threads = 0,
                                             const std::string& exec_provider =
                                                 "cpu");

}  // namespace crane
