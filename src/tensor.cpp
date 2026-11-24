#include "infercore/tensor.hpp"

#include <algorithm>
#include <cmath>

namespace infercore {

Tensor dequantize(const QTensor& q) {
  Tensor t(q.shape);
  const float s = q.scale;
  for (size_t i = 0; i < q.data.size(); ++i) t[i] = static_cast<float>(q.data[i]) * s;
  return t;
}

QTensor quantize_dynamic(const Tensor& t) {
  QTensor q;
  q.shape = t.shape();
  float absmax = 0.0f;
  for (size_t i = 0; i < t.numel(); ++i) absmax = std::max(absmax, std::fabs(t[i]));
  q.scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
  q.data.resize(t.numel());
  const float inv = 1.0f / q.scale;
  for (size_t i = 0; i < t.numel(); ++i) {
    const long v = std::lround(t[i] * inv);
    q.data[i] = static_cast<int8_t>(std::max<long>(-127, std::min<long>(127, v)));
  }
  return q;
}

QWeights quantize_weights_per_channel(const Tensor& w) {
  if (w.ndim() < 1) throw std::invalid_argument("quantize_weights: rank >= 1 required");
  QWeights q;
  q.shape = w.shape();
  const size_t channels = static_cast<size_t>(w.shape()[0]);
  const size_t per = w.numel() / channels;
  q.data.resize(w.numel());
  q.scales.resize(channels);
  for (size_t c = 0; c < channels; ++c) {
    float absmax = 0.0f;
    const float* base = w.data() + c * per;
    for (size_t i = 0; i < per; ++i) absmax = std::max(absmax, std::fabs(base[i]));
    q.scales[c] = absmax > 0.0f ? absmax / 127.0f : 1.0f;
    const float inv = 1.0f / q.scales[c];
    for (size_t i = 0; i < per; ++i) {
      const long v = std::lround(base[i] * inv);
      q.data[c * per + i] =
          static_cast<int8_t>(std::max<long>(-127, std::min<long>(127, v)));
    }
  }
  return q;
}

}  // namespace infercore
