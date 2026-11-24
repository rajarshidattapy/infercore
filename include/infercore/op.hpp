// InferCore - operator interface and implementations.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infercore/tensor.hpp"

namespace infercore {

class ThreadPool;

class Op {
 public:
  Op(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}
  virtual ~Op() = default;

  const std::string& name() const { return name_; }
  const std::string& type() const { return type_; }

  virtual Tensor forward(const std::vector<const Tensor*>& inputs, ThreadPool& pool) = 0;
  virtual QTensor forward_q(const std::vector<const QTensor*>& inputs, ThreadPool& pool) = 0;

  // Weighted ops override these; called by the model loader.
  virtual void set_weights(Tensor w, Tensor b) {
    (void)w; (void)b;
    throw std::runtime_error("op " + name_ + " does not take weights");
  }
  virtual void set_weights_q(QWeights w, Tensor b) {
    (void)w; (void)b;
    throw std::runtime_error("op " + name_ + " does not take quantized weights");
  }
  virtual bool needs_weights() const { return false; }
  virtual size_t weight_bytes_fp32() const { return 0; }
  virtual size_t weight_bytes_int8() const { return 0; }

 private:
  std::string name_;
  std::string type_;
};

std::unique_ptr<Op> make_conv2d(const std::string& name, int stride, int pad);
std::unique_ptr<Op> make_relu(const std::string& name);
std::unique_ptr<Op> make_maxpool(const std::string& name, int kernel, int stride);
std::unique_ptr<Op> make_flatten(const std::string& name);
std::unique_ptr<Op> make_gemm(const std::string& name);
std::unique_ptr<Op> make_softmax(const std::string& name);

}  // namespace infercore
