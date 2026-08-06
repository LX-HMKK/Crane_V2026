#include "inference/onnx_backend.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <opencv2/dnn.hpp>

#include "common/logging.hpp"

namespace crane {

namespace {

// Process-wide shared Ort::Env with a GLOBAL thread pool. Without this,
// every session creates its own intra-op pool (2 models × ncores threads
// oversubscribe the CPU, and the idle pool's spin-waiting steals cycles
// from the active one — measured at ~25% on a 32-core host).
Ort::Env& shared_env(int intra_threads) {
  static Ort::Env env = [&] {
    Ort::ThreadingOptions opts;
    opts.SetGlobalIntraOpNumThreads(intra_threads > 0 ? intra_threads : 0);
    opts.SetGlobalSpinControl(0);  // idle threads sleep, don't spin
    return Ort::Env(static_cast<OrtThreadingOptions*>(opts),
                    ORT_LOGGING_LEVEL_WARNING, "crane");
  }();
  return env;
}

}  // namespace

struct OnnxBackend::Impl {
  std::string exec_provider;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr input_name;
  Ort::AllocatedStringPtr output_name;
  std::vector<int64_t> input_dims;   // [1,3,H,W]
  std::vector<int64_t> output_dims;  // [1,4+nc,N]

  explicit Impl(const std::string& path, int intra_threads,
                const std::string& exec_provider)
      : exec_provider(exec_provider),
        session(make_session(shared_env(intra_threads), path, intra_threads,
                             exec_provider)),
        input_name(session.GetInputNameAllocated(0, allocator)),
        output_name(session.GetOutputNameAllocated(0, allocator)) {
    Ort::TypeInfo in_info = session.GetInputTypeInfo(0);
    input_dims = in_info.GetTensorTypeAndShapeInfo().GetShape();
    Ort::TypeInfo out_info = session.GetOutputTypeInfo(0);
    output_dims = out_info.GetTensorTypeAndShapeInfo().GetShape();
  }

 private:
  static Ort::Session make_session(Ort::Env& env, const std::string& path,
                                   int intra_threads,
                                   const std::string& exec_provider) {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    (void)intra_threads;  // global pool sizing is set on the shared env
    if (exec_provider == "cuda") {
      // Raw C API on purpose: the C++ wrapper (Ort::OrtCUDAProviderOptions) is
      // #ifdef'd out in CPU-package headers while the C struct and the
      // AppendExecutionProvider_CUDA function are always declared. So this
      // still compiles against the CPU package, and fails at session creation
      // below if the running onnxruntime.dll has no CUDA support.
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = 0;
      // The classic Windows footgun is a silent fallback to CPU when cuDNN /
      // cudart are missing or the wrong major version. Disable it: a CUDA
      // request that cannot be satisfied becomes a hard error instead.
      opts.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
      Ort::ThrowOnError(
          Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(opts, &cuda));
    } else if (exec_provider != "cpu") {
      throw std::runtime_error("Unknown execution provider \"" +
                               exec_provider + "\" (expected cpu or cuda)");
    }
    try {
      // ORTCHAR_T is wchar_t on Windows / char on POSIX; std::filesystem::path
      // exposes the same native char width via c_str(), so the model path is
      // accepted on both without a manual wide-char conversion.
      return Ort::Session(env, std::filesystem::path(path).c_str(), opts);
    } catch (const Ort::Exception& e) {
      throw std::runtime_error("Failed to load ONNX model " + path +
                               " (exec_provider=" + exec_provider + "): " +
                               e.what());
    }
  }
};

OnnxBackend::OnnxBackend(const std::string& onnx_path, int intra_threads,
                         const std::string& exec_provider)
    : impl_(std::make_unique<Impl>(onnx_path, intra_threads, exec_provider)) {
  if (impl_->input_dims.size() != 4 || impl_->input_dims[1] != 3) {
    throw std::runtime_error("Unexpected ONNX input shape for " + onnx_path);
  }
  if (impl_->output_dims.size() != 3) {
    throw std::runtime_error("Unexpected ONNX output shape for " + onnx_path);
  }
  meta_.imgsz_h = static_cast<int>(impl_->input_dims[2]);
  meta_.imgsz_w = static_cast<int>(impl_->input_dims[3]);
  meta_.nc = static_cast<int>(impl_->output_dims[1]) - 4;

  // Ultralytics custom metadata: names/stride/task.
  Ort::ModelMetadata md = impl_->session.GetModelMetadata();
  Ort::AllocatedStringPtr names_raw =
      md.LookupCustomMetadataMapAllocated("names", impl_->allocator);
  if (names_raw) {
    meta_.names = parse_ultralytics_names(names_raw.get());
  }
  if (static_cast<int>(meta_.names.size()) != meta_.nc) {
    // Fall back to synthetic names so zones still resolve deterministically.
    meta_.names.clear();
    for (int i = 0; i < meta_.nc; ++i) meta_.names[i] = std::to_string(i);
    LOG_WARN("ONNX metadata names missing/mismatch (nc=%d); using index "
             "names for %s",
             meta_.nc, onnx_path.c_str());
  }
  LOG_INFO("Loaded %s (onnxruntime/%s, %dx%d, nc=%d)", onnx_path.c_str(),
           impl_->exec_provider.c_str(), meta_.imgsz_w, meta_.imgsz_h, meta_.nc);
}

OnnxBackend::~OnnxBackend() = default;

RawOutputs OnnxBackend::run(const cv::Mat& letterboxed_bgr) {
  const int h = meta_.imgsz_h;
  const int w = meta_.imgsz_w;
  CV_Assert(letterboxed_bgr.rows == h && letterboxed_bgr.cols == w);

  // BGR → RGB, HWC → NCHW, uint8 → float32 /255.
  cv::Mat blob = cv::dnn::blobFromImage(letterboxed_bgr, 1.0 / 255.0, {w, h},
                                        cv::Scalar(), /*swapRB=*/true,
                                        /*crop=*/false);
  CV_Assert(blob.isContinuous());

  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t dims[4] = {1, 3, h, w};
  Ort::Value input = Ort::Value::CreateTensor<float>(
      mem_info, blob.ptr<float>(), blob.total(), dims, 4);

  const char* input_names[] = {impl_->input_name.get()};
  const char* output_names[] = {impl_->output_name.get()};
  auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names,
                                    &input, 1, output_names, 1);

  Ort::Value& out = outputs[0];
  auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
  const int rows = static_cast<int>(shape[1]);  // 4 + nc
  const int cols = static_cast<int>(shape[2]);  // N
  const float* data = out.GetTensorData<float>();

  RawOutputs raw;
  raw.layout = RawOutputs::Layout::kFused;
  raw.tensors.emplace_back(rows, cols, CV_32F);
  std::memcpy(raw.tensors[0].data, data, sizeof(float) * rows * cols);
  return raw;
}

std::unique_ptr<Yolo11Model> load_onnx_model(const std::string& onnx_path,
                                             float conf_thres,
                                             float iou_thres, int pad_value,
                                             int intra_threads,
                                             const std::string& exec_provider) {
  auto backend = std::make_unique<OnnxBackend>(onnx_path, intra_threads,
                                               exec_provider);
  ModelMeta meta = backend->meta();
  return std::make_unique<Yolo11Model>(std::move(backend), std::move(meta),
                                       conf_thres, iou_thres, pad_value);
}

}  // namespace crane
