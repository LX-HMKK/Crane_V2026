#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/types.hpp"

namespace crane {

// Raw model outputs in one of two layouts:
//  - kFused: single tensor [1, 4+nc, N] with DFL decoded in-graph and class
//    scores already sigmoid-activated (Ultralytics ONNX export).
//  - kHeads: 6 tensors in order [s_bbox, m_bbox, l_bbox, s_cls, m_cls, l_cls],
//    raw logits, bbox channels = 4*reg_max (RDK X5 BPU .bin export).
struct RawOutputs {
  enum class Layout { kFused, kHeads } layout = Layout::kFused;
  // kFused: one entry {4+nc, N} (row-major, as produced by the export;
  // transpose happens inside postprocess_fused).
  // kHeads: six entries; bbox {H*W, 4*reg_max}, cls {H*W, nc}.
  std::vector<cv::Mat> tensors;  // CV_32F, 2D row-major
};

// Inference backend interface: one model instance per object.
class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  // input: float32 NCHW [1,3,H,W] normalized 0..1 (fused backends)
  //        or whatever the backend declared via input_kind().
  virtual RawOutputs run(const cv::Mat& input) = 0;
  virtual RawOutputs::Layout output_layout() const = 0;
  virtual std::string name() const = 0;
};

// ── letterbox ───────────────────────────────────────────────────────────
struct LetterboxResult {
  cv::Mat image;
  double scale = 1.0;
  int pad_left = 0;
  int pad_top = 0;
};
// Mirrors yolo11_postprocess.letterbox (pad value configurable, default 127).
LetterboxResult letterbox(const cv::Mat& bgr, int new_h, int new_w,
                          int pad_value = 127);

// ── post-processing ─────────────────────────────────────────────────────
struct PostprocessConfig {
  int imgsz_h = 640;
  int imgsz_w = 640;
  int nc = 0;
  int reg_max = 16;
  std::vector<int> strides = {8, 16, 32};
  float conf_thres = 0.25f;
  float iou_thres = 0.45f;
};

// Fused ONNX output: dets in letterbox coordinates.
Detections postprocess_fused(const cv::Mat& output_2d, int nc,
                             float conf_thres, float iou_thres);

// 6-head BPU output, faithful port of yolo11_postprocess.postprocess
// (classification filter first, DFL decode only for survivors).
Detections postprocess_heads(const std::vector<cv::Mat>& heads,
                             const PostprocessConfig& cfg,
                             const cv::Mat& anchors,  // precomputed {A,2} CV_32F
                             const cv::Mat& dfl_proj  // {1,reg_max} CV_32F
);

cv::Mat make_anchors(int imgsz_h, int imgsz_w,
                     const std::vector<int>& strides);  // {A,2} CV_32F

// Map detections from letterbox space back to the original image.
void scale_coords(Detections& dets, double scale, int pad_left, int pad_top,
                  int img_width, int img_height);

// ── BGR → NV12 helper (RDK BPU input) ───────────────────────────────────
// Returns flat NV12 buffer (H*W*3/2 bytes); input must have even dims.
std::vector<uint8_t> nv12_from_bgr(const cv::Mat& bgr);

}  // namespace crane
