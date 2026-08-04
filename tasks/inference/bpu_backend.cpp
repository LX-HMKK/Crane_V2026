#include "inference/bpu_backend.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <dnn/hb_dnn.h>
#include <yaml-cpp/yaml.h>

#include "common/logging.hpp"

namespace crane {

namespace {

#define HB_CHECK(expr, what)                                              \
  do {                                                                    \
    int32_t rc_ = (expr);                                                 \
    if (rc_ != 0) {                                                       \
      throw std::runtime_error(std::string(what) + " failed, code " +     \
                               std::to_string(rc_));                      \
    }                                                                     \
  } while (0)

// Load the deployment yaml (beans.yaml / num.yaml) into ModelMeta.
ModelMeta load_bpu_cfg(const std::string& cfg_path) {
  YAML::Node cfg = YAML::LoadFile(cfg_path);
  ModelMeta meta;
  YAML::Node imgsz = cfg["imgsz"];
  if (imgsz.IsSequence()) {
    meta.imgsz_h = imgsz[0].as<int>();
    meta.imgsz_w = imgsz[1].as<int>();
  } else {
    meta.imgsz_h = meta.imgsz_w = imgsz.as<int>();
  }
  meta.nc = cfg["nc"].as<int>();
  meta.reg_max = cfg["reg_max"].as<int>(16);
  meta.strides = cfg["strides"].as<std::vector<int>>();
  YAML::Node names = cfg["names"];
  if (names.IsMap()) {
    for (const auto& kv : names) {
      meta.names[kv.first.as<int>()] = kv.second.as<std::string>();
    }
  } else if (names.IsSequence()) {
    for (size_t i = 0; i < names.size(); ++i) {
      meta.names[static_cast<int>(i)] = names[i].as<std::string>();
    }
  }
  if (static_cast<int>(meta.names.size()) != meta.nc) {
    throw std::runtime_error("BPU cfg " + cfg_path +
                             ": names count must equal nc");
  }
  return meta;
}

// Dequantize / convert one output tensor to a {H*W, C} CV_32F matrix.
cv::Mat tensor_to_mat(const hbDNNTensor& tensor) {
  const hbDNNTensorProperties& p = tensor.properties;
  // Expect [1, H, W, C] (NHWC) or [1, C, H, W] (NCHW) — handle NHWC, which
  // is what hb_mapper produces for these heads.
  if (p.validShape.numDimensions != 4) {
    throw std::runtime_error("BPU output with unexpected numDimensions " +
                             std::to_string(p.validShape.numDimensions));
  }
  int h, w, c;
  bool nchw = (p.tensorLayout == HB_DNN_LAYOUT_NCHW);
  if (nchw) {
    c = p.validShape.dimensionSize[1];
    h = p.validShape.dimensionSize[2];
    w = p.validShape.dimensionSize[3];
  } else {
    h = p.validShape.dimensionSize[1];
    w = p.validShape.dimensionSize[2];
    c = p.validShape.dimensionSize[3];
  }
  const int rows = h * w;

  // Element type size + dequantization params.
  const float* scale = nullptr;
  float scale_one = 1.0f;
  if (p.quantiType == SCALE && p.scale.scaleLen > 0) {
    scale = p.scale.scaleData;  // per-tensor (len 1) or per-channel (len C)
    if (p.scale.scaleLen == 1) {
      scale_one = p.scale.scaleData[0];
      scale = nullptr;
    }
  }

  cv::Mat out(rows, c, CV_32F);
  const int aligned_c = nchw ? c  // NCHW planes: stride is H*W, no chan pad
                             : (p.alignedShape.numDimensions == 4
                                    ? p.alignedShape.dimensionSize[3]
                                    : c);

  auto dequant_row = [&](const auto* src, int r, float* dst) {
    for (int ch = 0; ch < c; ++ch) {
      float s = scale ? scale[ch] : scale_one;
      dst[ch] = static_cast<float>(src[ch]) * s;
    }
  };

  const char* base = static_cast<const char*>(tensor.sysMem[0].virAddr);
  for (int r = 0; r < rows; ++r) {
    float* dst = out.ptr<float>(r);
    if (p.tensorType == HB_DNN_TENSOR_TYPE_F32) {
      const float* src =
          reinterpret_cast<const float*>(base) + static_cast<size_t>(r) * aligned_c;
      if (p.quantiType == SCALE || scale) {
        dequant_row(src, r, dst);
      } else {
        std::memcpy(dst, src, sizeof(float) * c);
      }
    } else if (p.tensorType == HB_DNN_TENSOR_TYPE_S8) {
      const int8_t* src =
          reinterpret_cast<const int8_t*>(base) + static_cast<size_t>(r) * aligned_c;
      dequant_row(src, r, dst);
    } else if (p.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(base) + static_cast<size_t>(r) * aligned_c;
      dequant_row(src, r, dst);
    } else if (p.tensorType == HB_DNN_TENSOR_TYPE_S32) {
      const int32_t* src =
          reinterpret_cast<const int32_t*>(base) + static_cast<size_t>(r) * aligned_c;
      dequant_row(src, r, dst);
    } else {
      throw std::runtime_error("BPU output with unsupported tensorType " +
                               std::to_string(p.tensorType));
    }
  }
  return out;
}

}  // namespace

struct BpuBackend::Impl {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNHandle_t dnn = nullptr;
  hbDNNTensor input{};
  std::vector<hbDNNTensor> outputs;
  std::vector<std::string> output_names;

  ~Impl() {
    if (packed) {
      for (auto& out : outputs) hbSysFreeMem(&out.sysMem[0]);
      hbSysFreeMem(&input.sysMem[0]);
      hbDNNRelease(packed);
    }
  }
};

BpuBackend::BpuBackend(const std::string& bin_path,
                       const std::string& cfg_path) {
  meta_ = load_bpu_cfg(cfg_path);
  impl_ = std::make_unique<Impl>();

  const char* files[] = {bin_path.c_str()};
  HB_CHECK(hbDNNInitializeFromFiles(&impl_->packed, files, 1),
           "hbDNNInitializeFromFiles(" + bin_path + ")");

  const char** name_list = nullptr;
  int32_t name_count = 0;
  HB_CHECK(hbDNNGetModelNameList(&name_list, &name_count, impl_->packed),
           "hbDNNGetModelNameList");
  if (name_count < 1) throw std::runtime_error("BPU model list empty");
  HB_CHECK(hbDNNGetModelHandle(&impl_->dnn, impl_->packed, name_list[0]),
           "hbDNNGetModelHandle");

  int32_t input_count = 0;
  HB_CHECK(hbDNNGetInputCount(&input_count, impl_->dnn), "hbDNNGetInputCount");
  if (input_count != 1) {
    throw std::runtime_error("BPU model input count " +
                             std::to_string(input_count) + " != 1");
  }
  HB_CHECK(
      hbDNNGetInputTensorProperties(&impl_->input.properties, impl_->dnn, 0),
      "hbDNNGetInputTensorProperties");
  if (impl_->input.properties.tensorType != HB_DNN_IMG_TYPE_NV12) {
    LOG_WARN("BPU input tensorType=%d (expected NV12=%d); conversion may be "
             "wrong",
             impl_->input.properties.tensorType, HB_DNN_IMG_TYPE_NV12);
  }
  HB_CHECK(hbSysAllocCachedMem(&impl_->input.sysMem[0],
                               impl_->input.properties.alignedByteSize),
           "hbSysAllocCachedMem(input)");

  int32_t output_count = 0;
  HB_CHECK(hbDNNGetOutputCount(&output_count, impl_->dnn),
           "hbDNNGetOutputCount");
  if (output_count != 6) {
    throw std::runtime_error("BPU model output count " +
                             std::to_string(output_count) + " != 6");
  }
  impl_->outputs.resize(output_count);
  impl_->output_names.resize(output_count);
  for (int32_t i = 0; i < output_count; ++i) {
    HB_CHECK(hbDNNGetOutputTensorProperties(&impl_->outputs[i].properties,
                                            impl_->dnn, i),
             "hbDNNGetOutputTensorProperties");
    HB_CHECK(hbSysAllocCachedMem(&impl_->outputs[i].sysMem[0],
                                 impl_->outputs[i].properties.alignedByteSize),
             "hbSysAllocCachedMem(output)");
    const char* out_name = nullptr;
    HB_CHECK(hbDNNGetOutputName(&out_name, impl_->dnn, i),
             "hbDNNGetOutputName");
    impl_->output_names[i] = out_name ? out_name : "";
    const auto& s = impl_->outputs[i].properties.validShape;
    LOG_INFO("BPU output %d '%s' dims=[%d %d %d %d] type=%d", i,
             impl_->output_names[i].c_str(), s.dimensionSize[0],
             s.dimensionSize[1], s.dimensionSize[2], s.dimensionSize[3],
             impl_->outputs[i].properties.tensorType);
  }
  available_ = true;
  LOG_INFO("Loaded %s (bpu-libdnn, %dx%d, nc=%d)", bin_path.c_str(),
           meta_.imgsz_w, meta_.imgsz_h, meta_.nc);
}

BpuBackend::~BpuBackend() = default;

RawOutputs BpuBackend::run(const cv::Mat& letterboxed_bgr) {
  // 1. BGR → flat NV12 into the input tensor.
  std::vector<uint8_t> nv12 = nv12_from_bgr(letterboxed_bgr);
  if (static_cast<int32_t>(nv12.size()) >
      impl_->input.properties.alignedByteSize) {
    throw std::runtime_error("NV12 buffer larger than BPU input allocation");
  }
  std::memcpy(impl_->input.sysMem[0].virAddr, nv12.data(), nv12.size());
  hbSysFlushMem(&impl_->input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

  // 2. Inference (synchronous wait).
  hbDNNInferCtrlParam ctrl;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM((&ctrl));
  hbDNNTaskHandle_t task = nullptr;
  hbDNNTensor* out_ptr = impl_->outputs.data();
  HB_CHECK(hbDNNInfer(&task, &out_ptr, &impl_->input, impl_->dnn, &ctrl),
           "hbDNNInfer");
  int32_t wait_rc = hbDNNWaitTaskDone(task, 1000);
  hbDNNReleaseTask(task);
  HB_CHECK(wait_rc, "hbDNNWaitTaskDone");

  // 3. Invalidate caches and convert to float matrices.
  std::vector<cv::Mat> mats;
  mats.reserve(impl_->outputs.size());
  for (auto& out : impl_->outputs) {
    hbSysFlushMem(&out.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    mats.push_back(tensor_to_mat(out));
  }

  // 4. Reorder into [s_bbox, m_bbox, l_bbox, s_cls, m_cls, l_cls] by shape:
  //    bbox heads have C = 4*reg_max; cls heads have C = nc; within each
  //    group, order by descending spatial size.
  const int bbox_c = 4 * meta_.reg_max;
  std::vector<int> bbox_idx, cls_idx;
  for (size_t i = 0; i < mats.size(); ++i) {
    if (mats[i].cols == bbox_c) {
      bbox_idx.push_back(static_cast<int>(i));
    } else if (mats[i].cols == meta_.nc) {
      cls_idx.push_back(static_cast<int>(i));
    } else {
      throw std::runtime_error("BPU output " + std::to_string(i) +
                               " channel count " +
                               std::to_string(mats[i].cols) +
                               " matches neither bbox nor cls");
    }
  }
  auto by_rows_desc = [&](int a, int b) { return mats[a].rows > mats[b].rows; };
  std::sort(bbox_idx.begin(), bbox_idx.end(), by_rows_desc);
  std::sort(cls_idx.begin(), cls_idx.end(), by_rows_desc);
  if (bbox_idx.size() != meta_.strides.size() ||
      cls_idx.size() != meta_.strides.size()) {
    throw std::runtime_error("BPU outputs do not match strides count");
  }

  RawOutputs raw;
  raw.layout = RawOutputs::Layout::kHeads;
  for (int i : bbox_idx) raw.tensors.push_back(std::move(mats[i]));
  for (int i : cls_idx) raw.tensors.push_back(std::move(mats[i]));
  return raw;
}

std::unique_ptr<Yolo11Model> load_bpu_model(const std::string& bin_path,
                                            const std::string& cfg_path,
                                            float conf_thres,
                                            float iou_thres, int pad_value) {
  auto backend = std::make_unique<BpuBackend>(bin_path, cfg_path);
  ModelMeta meta = backend->meta();
  return std::make_unique<Yolo11Model>(std::move(backend), std::move(meta),
                                       conf_thres, iou_thres, pad_value);
}

}  // namespace crane
