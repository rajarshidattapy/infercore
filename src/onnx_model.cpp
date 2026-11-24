#include "infercore/model.hpp"

#include <stdexcept>

#ifdef INFERCORE_WITH_ONNX

#include "onnxruntime_cxx_api.h"

namespace infercore {

namespace {

// On Windows, ORTCHAR_T is wchar_t and Ort::Session only accepts a wide path.
std::basic_string<ORTCHAR_T> to_ort_path(const std::string& path) {
#ifdef _WIN32
  return std::wstring(path.begin(), path.end());
#else
  return path;
#endif
}

class OnnxModel final : public IModel {
 public:
  explicit OnnxModel(const std::string& onnx_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "infercore"),
        session_(env_, to_ort_path(onnx_path).c_str(), Ort::SessionOptions{}) {
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session_.GetInputNameAllocated(0, alloc);
    auto out_name = session_.GetOutputNameAllocated(0, alloc);
    input_name_ = in_name.get();
    output_name_ = out_name.get();
    // Keep the TypeInfo objects alive while reading shapes: the
    // TensorTypeAndShapeInfo view is non-owning.
    auto in_type = session_.GetInputTypeInfo(0);
    auto out_type = session_.GetOutputTypeInfo(0);
    input_shape_ = in_type.GetTensorTypeAndShapeInfo().GetShape();
    output_shape_ = out_type.GetTensorTypeAndShapeInfo().GetShape();
    // Some exporters leave batch dim as -1; our models fix it to 1.
    for (auto& d : input_shape_)
      if (d < 0) d = 1;
    for (auto& d : output_shape_)
      if (d < 0) d = 1;
  }

  Tensor run(const Tensor& input) override {
    if (input.shape() != input_shape_)
      throw std::runtime_error("onnx: input shape mismatch");
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_v = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(input.data()), input.numel(), input_shape_.data(),
        input_shape_.size());
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{}, in_names, &input_v, 1, out_names, 1);
    Tensor out(output_shape_);
    float* src = outputs[0].GetTensorMutableData<float>();
    std::copy(src, src + out.numel(), out.data());
    return out;
  }

  std::vector<int64_t> output_shape() const override { return output_shape_; }
  std::vector<int64_t> input_shape() const override { return input_shape_; }
  std::string backend() const override { return "onnxruntime"; }
  std::string precision() const override { return "fp32"; }
  const std::vector<OpTiming>& last_timings() const override { return timings_; }

 private:
  Ort::Env env_;
  Ort::Session session_;
  std::string input_name_;
  std::string output_name_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<OpTiming> timings_;  // stays empty; ORT does not expose per-op here
};

}  // namespace

std::unique_ptr<IModel> load_onnx_model(const std::string& onnx_path) {
  return std::make_unique<OnnxModel>(onnx_path);
}

}  // namespace infercore

#else  // !INFERCORE_WITH_ONNX

namespace infercore {

std::unique_ptr<IModel> load_onnx_model(const std::string& onnx_path) {
  (void)onnx_path;
  throw std::runtime_error(
      "this build has no ONNX Runtime support (rebuild with -DINFERCORE_WITH_ONNX=ON "
      "and ONNXRUNTIME_ROOT set)");
}

}  // namespace infercore

#endif
