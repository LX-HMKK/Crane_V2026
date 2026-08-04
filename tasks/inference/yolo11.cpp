#include "yolo11.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace crane {

// Python round() is round-half-to-even; std::nearbyint with the default
// rounding mode matches it exactly (std::round is half-away-from-zero).
static inline int py_round(double v) {
  return static_cast<int>(std::nearbyint(v));
}

LetterboxResult letterbox(const cv::Mat& bgr, int new_h, int new_w,
                          int pad_value) {
  LetterboxResult out;
  const int h = bgr.rows;
  const int w = bgr.cols;
  double r = std::min(static_cast<double>(new_h) / h,
                      static_cast<double>(new_w) / w);
  out.scale = r;
  int new_unpad_w = py_round(w * r);
  int new_unpad_h = py_round(h * r);
  double dw = new_w - new_unpad_w;
  double dh = new_h - new_unpad_h;
  dw /= 2;
  dh /= 2;

  cv::Mat resized = bgr;
  if (w != new_unpad_w || h != new_unpad_h) {
    cv::resize(bgr, resized, {new_unpad_w, new_unpad_h}, 0, 0,
               cv::INTER_LINEAR);
  }
  int top = py_round(dh - 0.1);
  int bottom = py_round(dh + 0.1);
  int left = py_round(dw - 0.1);
  int right = py_round(dw + 0.1);
  cv::copyMakeBorder(resized, out.image, top, bottom, left, right,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(pad_value, pad_value, pad_value));
  out.pad_left = left;
  out.pad_top = top;
  return out;
}

cv::Mat make_anchors(int imgsz_h, int imgsz_w,
                     const std::vector<int>& strides) {
  int total = 0;
  for (int s : strides) total += (imgsz_h / s) * (imgsz_w / s);
  cv::Mat anchors(total, 2, CV_32F);
  float* ptr = anchors.ptr<float>();
  for (int s : strides) {
    int h = imgsz_h / s;
    int w = imgsz_w / s;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        *ptr++ = (x + 0.5f) * s;
        *ptr++ = (y + 0.5f) * s;
      }
    }
  }
  return anchors;
}

// Shared tail: class-aware NMS + descending-confidence sort.
// boxes/scores/cls_ids are parallel arrays in letterbox xyxy space.
static Detections nms_and_sort(const std::vector<cv::Rect2d>& boxes,
                               const std::vector<float>& scores,
                               const std::vector<int>& cls_ids,
                               float conf_thres, float iou_thres) {
  // Group candidate indices per class (class-aware NMS).
  std::vector<int> order(cls_ids.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return cls_ids[a] < cls_ids[b]; });

  std::vector<int> keep;
  size_t begin = 0;
  while (begin < order.size()) {
    size_t end = begin;
    while (end < order.size() && cls_ids[order[end]] == cls_ids[order[begin]])
      ++end;
    std::vector<cv::Rect2d> cls_boxes;
    std::vector<float> cls_scores;
    cls_boxes.reserve(end - begin);
    cls_scores.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      cls_boxes.push_back(boxes[order[i]]);
      cls_scores.push_back(scores[order[i]]);
    }
    std::vector<int> selected;
    cv::dnn::NMSBoxes(cls_boxes, cls_scores, conf_thres, iou_thres, selected);
    for (int idx : selected) keep.push_back(order[begin + idx]);
    begin = end;
  }

  std::sort(keep.begin(), keep.end(),
            [&](int a, int b) { return scores[a] > scores[b]; });

  Detections dets;
  dets.reserve(keep.size());
  for (int idx : keep) {
    Detection d;
    d.x1 = static_cast<float>(boxes[idx].x);
    d.y1 = static_cast<float>(boxes[idx].y);
    d.x2 = static_cast<float>(boxes[idx].x + boxes[idx].width);
    d.y2 = static_cast<float>(boxes[idx].y + boxes[idx].height);
    d.conf = scores[idx];
    d.cls_id = cls_ids[idx];
    dets.push_back(d);
  }
  return dets;
}

Detections postprocess_fused(const cv::Mat& output_2d, int nc,
                             float conf_thres, float iou_thres) {
  CV_Assert(output_2d.type() == CV_32F);
  CV_Assert(output_2d.rows == 4 + nc);
  // Transpose to {N, 4+nc} so each candidate is one contiguous row.
  cv::Mat preds;
  cv::transpose(output_2d, preds);
  const int n = preds.rows;
  const int cols = preds.cols;  // 4 + nc

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  std::vector<int> cls_ids;
  boxes.reserve(64);
  scores.reserve(64);
  cls_ids.reserve(64);

  for (int i = 0; i < n; ++i) {
    const float* row = preds.ptr<float>(i);
    const float* cls = row + 4;
    int best_cls = 0;
    float best_score = cls[0];
    for (int c = 1; c < nc; ++c) {
      if (cls[c] > best_score) {
        best_score = cls[c];
        best_cls = c;
      }
    }
    if (best_score < conf_thres) continue;
    float cx = row[0], cy = row[1], w = row[2], h = row[3];
    boxes.emplace_back(cx - w * 0.5, cy - h * 0.5, w, h);
    scores.push_back(best_score);
    cls_ids.push_back(best_cls);
    (void)cols;
  }
  return nms_and_sort(boxes, scores, cls_ids, conf_thres, iou_thres);
}

Detections postprocess_heads(const std::vector<cv::Mat>& heads,
                             const PostprocessConfig& cfg,
                             const cv::Mat& anchors,
                             const cv::Mat& dfl_proj) {
  CV_Assert(heads.size() == 6);
  const int nc = cfg.nc;
  const int reg_max = cfg.reg_max;
  const size_t num_heads = cfg.strides.size();

  // ── classification: sigmoid + argmax over the concatenated heads ────
  // Per-head candidate count and running mask, preserving head order.
  std::vector<int> head_counts(num_heads);
  int total = 0;
  for (size_t i = 0; i < num_heads; ++i) {
    CV_Assert(heads[i + 3].type() == CV_32F && heads[i + 3].cols == nc);
    head_counts[i] = heads[i + 3].rows;
    total += head_counts[i];
  }

  std::vector<float> conf(total);
  std::vector<int> cls_id(total);
  int idx = 0;
  for (size_t h = 0; h < num_heads; ++h) {
    const cv::Mat& cls_head = heads[h + 3];
    for (int r = 0; r < head_counts[h]; ++r, ++idx) {
      const float* row = cls_head.ptr<float>(r);
      // sigmoid inline (numpy: 1/(1+exp(-x)) in float32)
      float best = 1.0f / (1.0f + std::exp(-row[0]));
      int best_c = 0;
      for (int c = 1; c < nc; ++c) {
        float s = 1.0f / (1.0f + std::exp(-row[c]));
        if (s > best) {
          best = s;
          best_c = c;
        }
      }
      conf[idx] = best;
      cls_id[idx] = best_c;
    }
  }

  // ── threshold mask → DFL decode survivors only ──────────────────────
  std::vector<int> cand;  // global anchor indices that passed
  cand.reserve(256);
  for (int i = 0; i < total; ++i) {
    if (conf[i] >= cfg.conf_thres) cand.push_back(i);
  }
  if (cand.empty()) return {};

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  std::vector<int> cls_ids;
  boxes.reserve(cand.size());
  scores.reserve(cand.size());
  cls_ids.reserve(cand.size());

  const float* anchor_ptr = anchors.ptr<float>();
  const float* proj = dfl_proj.ptr<float>();

  for (int g : cand) {
    // Locate the bbox head + row for this global index.
    int offset = g;
    size_t h = 0;
    while (h < num_heads && offset >= head_counts[h]) {
      offset -= head_counts[h];
      ++h;
    }
    const float* bbox_row = heads[h].ptr<float>(offset);

    // DFL decode: softmax over reg_max bins per side, weighted by proj.
    float ltrb[4];
    for (int side = 0; side < 4; ++side) {
      const float* logits = bbox_row + side * reg_max;
      float max_logit = *std::max_element(logits, logits + reg_max);
      float sum = 0.0f;
      float weighted = 0.0f;
      for (int b = 0; b < reg_max; ++b) {
        float e = std::exp(logits[b] - max_logit);
        sum += e;
        weighted += e * proj[b];
      }
      ltrb[side] = weighted / sum;
    }
    float ax = anchor_ptr[g * 2];
    float ay = anchor_ptr[g * 2 + 1];
    double x1 = ax - ltrb[0], y1 = ay - ltrb[1];
    double x2 = ax + ltrb[2], y2 = ay + ltrb[3];
    boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
    scores.push_back(conf[g]);
    cls_ids.push_back(cls_id[g]);
  }

  return nms_and_sort(boxes, scores, cls_ids, cfg.conf_thres, cfg.iou_thres);
}

void scale_coords(Detections& dets, double scale, int pad_left, int pad_top,
                  int img_width, int img_height) {
  for (auto& d : dets) {
    d.x1 = static_cast<float>((d.x1 - pad_left) / scale);
    d.x2 = static_cast<float>((d.x2 - pad_left) / scale);
    d.y1 = static_cast<float>((d.y1 - pad_top) / scale);
    d.y2 = static_cast<float>((d.y2 - pad_top) / scale);
    d.x1 = std::min(std::max(d.x1, 0.0f), static_cast<float>(img_width));
    d.x2 = std::min(std::max(d.x2, 0.0f), static_cast<float>(img_width));
    d.y1 = std::min(std::max(d.y1, 0.0f), static_cast<float>(img_height));
    d.y2 = std::min(std::max(d.y2, 0.0f), static_cast<float>(img_height));
  }
}

std::vector<uint8_t> nv12_from_bgr(const cv::Mat& bgr) {
  const int h = bgr.rows;
  const int w = bgr.cols;
  if (h % 2 != 0 || w % 2 != 0) {
    throw std::invalid_argument("NV12 conversion requires even dimensions, " +
                                std::to_string(h) + "x" + std::to_string(w));
  }
  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);  // H*1.5 x W, 1 channel
  const uint8_t* src = i420.ptr<uint8_t>();
  const size_t area = static_cast<size_t>(h) * w;
  std::vector<uint8_t> nv12(area * 3 / 2);
  // Y plane
  std::copy(src, src + area, nv12.begin());
  // Interleave U/V from the two quarter planes.
  const uint8_t* u = src + area;
  const uint8_t* v = u + area / 4;
  uint8_t* uv = nv12.data() + area;
  for (size_t i = 0; i < area / 4; ++i) {
    uv[i * 2] = u[i];
    uv[i * 2 + 1] = v[i];
  }
  return nv12;
}

}  // namespace crane
