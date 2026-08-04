#include "inference/model.hpp"

#include <stdexcept>

#include "common/logging.hpp"

namespace crane {

Yolo11Model::Yolo11Model(std::unique_ptr<InferenceBackend> backend,
                         ModelMeta meta, float conf_thres, float iou_thres,
                         int pad_value)
    : backend_(std::move(backend)),
      meta_(std::move(meta)),
      pad_value_(pad_value),
      backend_name_(backend_->name()),
      conf_thres_(conf_thres),
      iou_thres_(iou_thres) {
  if (meta_.nc <= 0) {
    throw std::invalid_argument("ModelMeta.nc must be positive");
  }
  if (backend_->output_layout() == RawOutputs::Layout::kHeads) {
    anchors_ = make_anchors(meta_.imgsz_h, meta_.imgsz_w, meta_.strides);
    dfl_proj_ = cv::Mat(1, meta_.reg_max, CV_32F);
    for (int i = 0; i < meta_.reg_max; ++i) {
      dfl_proj_.at<float>(0, i) = static_cast<float>(i);
    }
  }
}

Detections Yolo11Model::predict(const cv::Mat& bgr) {
  CV_Assert(!bgr.empty() && bgr.channels() == 3);

  float conf_thres, iou_thres;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    conf_thres = conf_thres_;
    iou_thres = iou_thres_;
  }

  LetterboxResult lb =
      letterbox(bgr, meta_.imgsz_h, meta_.imgsz_w, pad_value_);
  RawOutputs raw = backend_->run(lb.image);

  Detections dets;
  if (raw.layout == RawOutputs::Layout::kFused) {
    CV_Assert(raw.tensors.size() == 1);
    dets = postprocess_fused(raw.tensors[0], meta_.nc, conf_thres, iou_thres);
  } else {
    PostprocessConfig cfg;
    cfg.imgsz_h = meta_.imgsz_h;
    cfg.imgsz_w = meta_.imgsz_w;
    cfg.nc = meta_.nc;
    cfg.reg_max = meta_.reg_max;
    cfg.strides = meta_.strides;
    cfg.conf_thres = conf_thres;
    cfg.iou_thres = iou_thres;
    dets = postprocess_heads(raw.tensors, cfg, anchors_, dfl_proj_);
  }
  scale_coords(dets, lb.scale, lb.pad_left, lb.pad_top, bgr.cols, bgr.rows);
  return dets;
}

void Yolo11Model::set_conf_thres(float v) {
  std::lock_guard<std::mutex> lock(mutex_);
  conf_thres_ = v;
}

float Yolo11Model::conf_thres() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return conf_thres_;
}

void Yolo11Model::set_iou_thres(float v) {
  std::lock_guard<std::mutex> lock(mutex_);
  iou_thres_ = v;
}

float Yolo11Model::iou_thres() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return iou_thres_;
}

std::map<int, std::string> parse_ultralytics_names(const std::string& raw) {
  std::map<int, std::string> names;
  size_t begin = raw.find('{');
  size_t end = raw.rfind('}');
  if (begin == std::string::npos || end == std::string::npos || end <= begin) {
    return names;
  }
  std::string body = raw.substr(begin + 1, end - begin - 1);
  size_t pos = 0;
  while (pos < body.size()) {
    // key: integer up to ':'
    size_t colon = body.find(':', pos);
    if (colon == std::string::npos) break;
    int key;
    try {
      key = std::stoi(body.substr(pos, colon - pos));
    } catch (...) {
      break;
    }
    // value: 'quoted' or "quoted"
    size_t q1 = body.find_first_of("'\"", colon);
    if (q1 == std::string::npos) break;
    char quote = body[q1];
    size_t q2 = body.find(quote, q1 + 1);
    if (q2 == std::string::npos) break;
    names[key] = body.substr(q1 + 1, q2 - q1 - 1);
    // advance past the separating comma before the next key
    size_t comma = body.find(',', q2);
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return names;
}

}  // namespace crane
