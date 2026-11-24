// InferCore - tensor types (fp32 and quantized int8).
#pragma once

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace infercore {

// Dense fp32 tensor, row-major, NCHW for images.
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
    data_.resize(numel(), 0.0f);
  }
  Tensor(std::vector<int64_t> shape, std::vector<float> data)
      : shape_(std::move(shape)), data_(std::move(data)) {
    if (data_.size() != numel())
      throw std::invalid_argument("Tensor: data size does not match shape");
  }

  const std::vector<int64_t>& shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t numel() const {
    if (shape_.empty()) return data_.size();
    size_t n = 1;
    for (int64_t d : shape_) n *= static_cast<size_t>(d);
    return n;
  }

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }
  float& operator[](size_t i) { return data_[i]; }
  float operator[](size_t i) const { return data_[i]; }

  void reshape(std::vector<int64_t> shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    if (n != numel()) throw std::invalid_argument("reshape: element count changed");
    shape_ = std::move(shape);
  }

 private:
  std::vector<int64_t> shape_;
  std::vector<float> data_;
};

// Dynamically quantized int8 tensor: symmetric, per-tensor scale, zero point 0.
struct QTensor {
  std::vector<int64_t> shape;
  std::vector<int8_t> data;
  float scale = 1.0f;

  size_t numel() const {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
  }
};

// Quantized weights: per-output-channel scales (channel dim = dim 0).
struct QWeights {
  std::vector<int64_t> shape;
  std::vector<int8_t> data;
  std::vector<float> scales;  // one per output channel
};

Tensor dequantize(const QTensor& q);
QTensor quantize_dynamic(const Tensor& t);  // absmax scale, zero point 0
QWeights quantize_weights_per_channel(const Tensor& w);  // dim 0 = out channels

}  // namespace infercore
