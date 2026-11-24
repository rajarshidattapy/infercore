#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "infercore/kernels.hpp"
#include "infercore/op.hpp"
#include "infercore/thread_pool.hpp"

namespace infercore {

namespace {

const Tensor& single(const std::vector<const Tensor*>& inputs, const char* op) {
  if (inputs.size() != 1) throw std::runtime_error(std::string(op) + ": expects 1 input");
  return *inputs[0];
}
const QTensor& single_q(const std::vector<const QTensor*>& inputs, const char* op) {
  if (inputs.size() != 1) throw std::runtime_error(std::string(op) + ": expects 1 input");
  return *inputs[0];
}

class Conv2DOp final : public Op {
 public:
  Conv2DOp(const std::string& name, int stride, int pad)
      : Op(name, "Conv2D"), stride_(stride), pad_(pad) {}

  bool needs_weights() const override { return true; }
  size_t weight_bytes_fp32() const override { return w_.numel() * 4 + b_.numel() * 4; }
  size_t weight_bytes_int8() const override {
    return qw_.data.size() + qw_.scales.size() * 4 + b_.numel() * 4;
  }

  void set_weights(Tensor w, Tensor b) override {
    if (w.ndim() != 4) throw std::runtime_error(name() + ": weights must be [OC,C,kH,kW]");
    w_ = std::move(w);
    b_ = std::move(b);
    qw_ = quantize_weights_per_channel(w_);
  }
  void set_weights_q(QWeights w, Tensor b) override {
    if (w.shape.size() != 4)
      throw std::runtime_error(name() + ": weights must be [OC,C,kH,kW]");
    qw_ = std::move(w);
    b_ = std::move(b);
  }

  Tensor forward(const std::vector<const Tensor*>& inputs, ThreadPool& pool) override {
    const Tensor& in = single(inputs, "Conv2D");
    if (in.ndim() != 4 || in.shape()[0] != 1)
      throw std::runtime_error(name() + ": input must be [1,C,H,W]");
    const int C = static_cast<int>(in.shape()[1]);
    const int H = static_cast<int>(in.shape()[2]);
    const int W = static_cast<int>(in.shape()[3]);
    const int OC = static_cast<int>(w_.shape()[0]);
    const int kH = static_cast<int>(w_.shape()[2]);
    const int kW = static_cast<int>(w_.shape()[3]);
    const int outH = (H + 2 * pad_ - kH) / stride_ + 1;
    const int outW = (W + 2 * pad_ - kW) / stride_ + 1;
    const int P = outH * outW;
    const int K = C * kH * kW;
    std::vector<float> col(static_cast<size_t>(P) * K);
    im2col_f32(in.data(), C, H, W, kH, kW, stride_, pad_, outH, outW, col.data());
    Tensor out({1, OC, outH, outW});
    // out[OC, P] = W[OC, K] * col[P, K]^T
    gemm_f32(w_.data(), col.data(), out.data(), OC, P, K, /*transB=*/true, &pool);
    // add bias
    for (int oc = 0; oc < OC; ++oc) {
      float* row = out.data() + static_cast<size_t>(oc) * P;
      const float bv = b_.numel() ? b_[oc] : 0.0f;
      for (int p = 0; p < P; ++p) row[p] += bv;
    }
    return out;
  }

  QTensor forward_q(const std::vector<const QTensor*>& inputs, ThreadPool& pool) override {
    const QTensor& in = single_q(inputs, "Conv2D");
    if (in.shape.size() != 4 || in.shape[0] != 1)
      throw std::runtime_error(name() + ": input must be [1,C,H,W]");
    const int C = static_cast<int>(in.shape[1]);
    const int H = static_cast<int>(in.shape[2]);
    const int W = static_cast<int>(in.shape[3]);
    const int OC = static_cast<int>(qw_.shape[0]);
    const int kH = static_cast<int>(qw_.shape[2]);
    const int kW = static_cast<int>(qw_.shape[3]);
    const int outH = (H + 2 * pad_ - kH) / stride_ + 1;
    const int outW = (W + 2 * pad_ - kW) / stride_ + 1;
    const int P = outH * outW;
    const int K = C * kH * kW;
    std::vector<int8_t> col(static_cast<size_t>(P) * K);
    im2col_s8(in.data.data(), C, H, W, kH, kW, stride_, pad_, outH, outW, col.data());
    // acc[P, OC] = col[P, K] * W[OC, K]^T
    std::vector<int32_t> acc(static_cast<size_t>(P) * OC);
    gemm_s8s8s32(col.data(), qw_.data.data(), acc.data(), P, OC, K, &pool);
    // dequantize to fp32, add bias, then dynamically requantize the activations
    Tensor f({1, OC, outH, outW});
    for (int oc = 0; oc < OC; ++oc) {
      const float s = in.scale * qw_.scales[oc];
      const float bv = b_.numel() ? b_[oc] : 0.0f;
      float* row = f.data() + static_cast<size_t>(oc) * P;
      for (int p = 0; p < P; ++p)
        row[p] = static_cast<float>(acc[static_cast<size_t>(p) * OC + oc]) * s + bv;
    }
    return quantize_dynamic(f);
  }

 private:
  int stride_;
  int pad_;
  Tensor w_;
  Tensor b_;
  QWeights qw_;
};

class GemmOp final : public Op {
 public:
  explicit GemmOp(const std::string& name) : Op(name, "Gemm") {}

  bool needs_weights() const override { return true; }
  size_t weight_bytes_fp32() const override { return w_.numel() * 4 + b_.numel() * 4; }
  size_t weight_bytes_int8() const override {
    return qw_.data.size() + qw_.scales.size() * 4 + b_.numel() * 4;
  }

  void set_weights(Tensor w, Tensor b) override {
    if (w.ndim() != 2) throw std::runtime_error(name() + ": weights must be [N,K]");
    w_ = std::move(w);
    b_ = std::move(b);
    qw_ = quantize_weights_per_channel(w_);
  }
  void set_weights_q(QWeights w, Tensor b) override {
    if (w.shape.size() != 2) throw std::runtime_error(name() + ": weights must be [N,K]");
    qw_ = std::move(w);
    b_ = std::move(b);
  }

  Tensor forward(const std::vector<const Tensor*>& inputs, ThreadPool& pool) override {
    const Tensor& in = single(inputs, "Gemm");
    const size_t K = in.numel();
    const size_t N = static_cast<size_t>(w_.shape()[0]);
    Tensor out({1, static_cast<int64_t>(N)});
    gemm_f32(in.data(), w_.data(), out.data(), 1, N, K, /*transB=*/true, &pool);
    for (size_t n = 0; n < N; ++n) out[n] += b_.numel() ? b_[n] : 0.0f;
    return out;
  }

  QTensor forward_q(const std::vector<const QTensor*>& inputs, ThreadPool& pool) override {
    const QTensor& in = single_q(inputs, "Gemm");
    const size_t K = in.numel();
    const size_t N = static_cast<size_t>(qw_.shape[0]);
    std::vector<int32_t> acc(N);
    gemm_s8s8s32(in.data.data(), qw_.data.data(), acc.data(), 1, N, K, &pool);
    Tensor f({1, static_cast<int64_t>(N)});
    for (size_t n = 0; n < N; ++n) {
      const float s = in.scale * qw_.scales[n];
      f[n] = static_cast<float>(acc[n]) * s + (b_.numel() ? b_[n] : 0.0f);
    }
    return quantize_dynamic(f);
  }

 private:
  Tensor w_;   // [N, K]
  Tensor b_;   // [N]
  QWeights qw_;
};

class ReLUOp final : public Op {
 public:
  explicit ReLUOp(const std::string& name) : Op(name, "ReLU") {}
  Tensor forward(const std::vector<const Tensor*>& inputs, ThreadPool&) override {
    Tensor out = single(inputs, "ReLU");
    for (size_t i = 0; i < out.numel(); ++i) out[i] = std::max(0.0f, out[i]);
    return out;
  }
  QTensor forward_q(const std::vector<const QTensor*>& inputs, ThreadPool&) override {
    QTensor out = single_q(inputs, "ReLU");
    for (auto& v : out.data) v = std::max<int8_t>(v, 0);  // symmetric: 0 stays 0
    return out;
  }
};

class MaxPoolOp final : public Op {
 public:
  MaxPoolOp(const std::string& name, int kernel, int stride)
      : Op(name, "MaxPool"), k_(kernel), s_(stride) {}

  Tensor forward(const std::vector<const Tensor*>& inputs, ThreadPool&) override {
    const Tensor& in = single(inputs, "MaxPool");
    const int C = static_cast<int>(in.shape()[1]);
    const int H = static_cast<int>(in.shape()[2]);
    const int W = static_cast<int>(in.shape()[3]);
    const int outH = (H - k_) / s_ + 1;
    const int outW = (W - k_) / s_ + 1;
    Tensor out({1, C, outH, outW});
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < outH; ++oh)
        for (int ow = 0; ow < outW; ++ow) {
          float best = -std::numeric_limits<float>::infinity();
          for (int kh = 0; kh < k_; ++kh)
            for (int kw = 0; kw < k_; ++kw) {
              const float v = in[(static_cast<size_t>(c) * H + oh * s_ + kh) * W +
                                 ow * s_ + kw];
              best = std::max(best, v);
            }
          out[(static_cast<size_t>(c) * outH + oh) * outW + ow] = best;
        }
    return out;
  }

  QTensor forward_q(const std::vector<const QTensor*>& inputs, ThreadPool&) override {
    const QTensor& in = single_q(inputs, "MaxPool");
    const int C = static_cast<int>(in.shape[1]);
    const int H = static_cast<int>(in.shape[2]);
    const int W = static_cast<int>(in.shape[3]);
    const int outH = (H - k_) / s_ + 1;
    const int outW = (W - k_) / s_ + 1;
    QTensor out;
    out.shape = {1, C, outH, outW};
    out.scale = in.scale;  // max preserves scale
    out.data.resize(static_cast<size_t>(C) * outH * outW);
    for (int c = 0; c < C; ++c)
      for (int oh = 0; oh < outH; ++oh)
        for (int ow = 0; ow < outW; ++ow) {
          int8_t best = -128;
          for (int kh = 0; kh < k_; ++kh)
            for (int kw = 0; kw < k_; ++kw)
              best = std::max(best, in.data[(static_cast<size_t>(c) * H + oh * s_ + kh) *
                                                W + ow * s_ + kw]);
          out.data[(static_cast<size_t>(c) * outH + oh) * outW + ow] = best;
        }
    return out;
  }

 private:
  int k_;
  int s_;
};

class FlattenOp final : public Op {
 public:
  explicit FlattenOp(const std::string& name) : Op(name, "Flatten") {}
  Tensor forward(const std::vector<const Tensor*>& inputs, ThreadPool&) override {
    Tensor out = single(inputs, "Flatten");
    out.reshape({1, static_cast<int64_t>(out.numel())});
    return out;
  }
  QTensor forward_q(const std::vector<const QTensor*>& inputs, ThreadPool&) override {
    QTensor out = single_q(inputs, "Flatten");
    out.shape = {1, static_cast<int64_t>(out.numel())};
    return out;
  }
};

class SoftmaxOp final : public Op {
 public:
  explicit SoftmaxOp(const std::string& name) : Op(name, "Softmax") {}

  Tensor forward(const std::vector<const Tensor*>& inputs, ThreadPool&) override {
    Tensor out = single(inputs, "Softmax");
    softmax_inplace(out.data(), out.numel());
    return out;
  }
  QTensor forward_q(const std::vector<const QTensor*>& inputs, ThreadPool&) override {
    // Softmax is computed in fp32 after dequantization; argmax is preserved by
    // the monotonic requantization.
    Tensor f = dequantize(single_q(inputs, "Softmax"));
    softmax_inplace(f.data(), f.numel());
    return quantize_dynamic(f);
  }

 private:
  static void softmax_inplace(float* x, size_t n) {
    const float mx = *std::max_element(x, x + n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      x[i] = std::exp(x[i] - mx);
      sum += x[i];
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
  }
};

}  // namespace

std::unique_ptr<Op> make_conv2d(const std::string& name, int stride, int pad) {
  return std::make_unique<Conv2DOp>(name, stride, pad);
}
std::unique_ptr<Op> make_relu(const std::string& name) {
  return std::make_unique<ReLUOp>(name);
}
std::unique_ptr<Op> make_maxpool(const std::string& name, int kernel, int stride) {
  return std::make_unique<MaxPoolOp>(name, kernel, stride);
}
std::unique_ptr<Op> make_flatten(const std::string& name) {
  return std::make_unique<FlattenOp>(name);
}
std::unique_ptr<Op> make_gemm(const std::string& name) {
  return std::make_unique<GemmOp>(name);
}
std::unique_ptr<Op> make_softmax(const std::string& name) {
  return std::make_unique<SoftmaxOp>(name);
}

}  // namespace infercore
