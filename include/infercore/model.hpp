// InferCore - model loading and execution.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infercore/tensor.hpp"

namespace infercore {

enum class Precision { kFp32, kInt8 };

struct OpTiming {
  std::string op_name;
  std::string op_type;
  double micros = 0.0;
};

class IModel {
 public:
  virtual ~IModel() = default;
  virtual Tensor run(const Tensor& input) = 0;
  virtual std::vector<int64_t> output_shape() const = 0;
  virtual std::vector<int64_t> input_shape() const = 0;
  virtual std::string backend() const = 0;
  virtual std::string precision() const = 0;
  // Per-op timings of the most recent run (native backend only; empty for ONNX).
  virtual const std::vector<OpTiming>& last_timings() const = 0;
};

// graph_json: model graph description; weights: ICW1 (fp32) or ICQ1 (int8).
std::unique_ptr<IModel> load_native_model(const std::string& graph_json,
                                          const std::string& weights_path,
                                          Precision precision, unsigned num_threads);

// ONNX Runtime backend. Throws at load time if the build has no ONNX support.
std::unique_ptr<IModel> load_onnx_model(const std::string& onnx_path);

}  // namespace infercore
