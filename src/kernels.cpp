#include "infercore/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "infercore/thread_pool.hpp"

namespace infercore {

namespace {

inline size_t pick_grain(size_t M, size_t inner_cost) {
  // Aim for chunks of roughly 256k multiply-adds, at least 1 row.
  size_t rows = 262144 / std::max<size_t>(inner_cost, 1);
  return std::max<size_t>(rows, 1);
}

}  // namespace

void gemm_f32(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
              bool transB, ThreadPool* pool) {
  auto rows = [&](size_t lo, size_t hi) {
    for (size_t m = lo; m < hi; ++m) {
      const float* a = A + m * K;
      float* c = C + m * N;
      if (transB) {
        for (size_t n = 0; n < N; ++n) {
          const float* b = B + n * K;
          float acc = 0.0f;
          size_t k = 0;
          for (; k + 4 <= K; k += 4)
            acc += a[k] * b[k] + a[k + 1] * b[k + 1] + a[k + 2] * b[k + 2] +
                   a[k + 3] * b[k + 3];
          for (; k < K; ++k) acc += a[k] * b[k];
          c[n] = acc;
        }
      } else {
        std::fill(c, c + N, 0.0f);
        for (size_t k = 0; k < K; ++k) {
          const float av = a[k];
          const float* b = B + k * N;
          for (size_t n = 0; n < N; ++n) c[n] += av * b[n];
        }
      }
    }
  };
  if (pool)
    pool->parallel_for(M, pick_grain(M, N * K), rows);
  else
    rows(0, M);
}

void gemm_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C, size_t M, size_t N,
                  size_t K, ThreadPool* pool) {
  auto rows = [&](size_t lo, size_t hi) {
    for (size_t m = lo; m < hi; ++m) {
      const int8_t* a = A + m * K;
      int32_t* c = C + m * N;
      for (size_t n = 0; n < N; ++n) {
        const int8_t* b = B + n * K;
        int32_t acc = 0;
        size_t k = 0;
        for (; k + 8 <= K; k += 8) {
          int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
          s0 += a[k] * b[k] + a[k + 1] * b[k + 1];
          s1 += a[k + 2] * b[k + 2] + a[k + 3] * b[k + 3];
          s2 += a[k + 4] * b[k + 4] + a[k + 5] * b[k + 5];
          s3 += a[k + 6] * b[k + 6] + a[k + 7] * b[k + 7];
          acc += (s0 + s1) + (s2 + s3);
        }
        for (; k < K; ++k) acc += static_cast<int32_t>(a[k]) * b[k];
        c[n] = acc;
      }
    }
  };
  if (pool)
    pool->parallel_for(M, pick_grain(M, N * K), rows);
  else
    rows(0, M);
}

template <typename T>
static void im2col_impl(const T* in, int C, int H, int W, int kH, int kW, int stride,
                        int pad, int outH, int outW, T* col) {
  const int K = C * kH * kW;
  for (int p = 0; p < outH * outW; ++p) {
    const int oh = p / outW;
    const int ow = p % outW;
    T* row = col + static_cast<size_t>(p) * K;
    int idx = 0;
    for (int c = 0; c < C; ++c) {
      for (int kh = 0; kh < kH; ++kh) {
        const int ih = oh * stride + kh - pad;
        for (int kw = 0; kw < kW; ++kw) {
          const int iw = ow * stride + kw - pad;
          const bool inside = (ih >= 0 && ih < H && iw >= 0 && iw < W);
          row[idx++] = inside ? in[(static_cast<size_t>(c) * H + ih) * W + iw]
                              : static_cast<T>(0);
        }
      }
    }
  }
}

void im2col_f32(const float* in, int C, int H, int W, int kH, int kW, int stride,
                int pad, int outH, int outW, float* col) {
  im2col_impl(in, C, H, W, kH, kW, stride, pad, outH, outW, col);
}

void im2col_s8(const int8_t* in, int C, int H, int W, int kH, int kW, int stride,
               int pad, int outH, int outW, int8_t* col) {
  im2col_impl(in, C, H, W, kH, kW, stride, pad, outH, outW, col);
}

void conv2d_f32_reference(const float* in, const float* w, const float* bias, int C,
                          int H, int Wd, int OC, int kH, int kW, int stride, int pad,
                          float* out) {
  const int outH = (H + 2 * pad - kH) / stride + 1;
  const int outW = (Wd + 2 * pad - kW) / stride + 1;
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < outH; ++oh) {
      for (int ow = 0; ow < outW; ++ow) {
        float acc = bias ? bias[oc] : 0.0f;
        for (int c = 0; c < C; ++c) {
          for (int kh = 0; kh < kH; ++kh) {
            const int ih = oh * stride + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
              const int iw = ow * stride + kw - pad;
              if (iw < 0 || iw >= Wd) continue;
              acc += in[(static_cast<size_t>(c) * H + ih) * Wd + iw] *
                     w[((static_cast<size_t>(oc) * C + c) * kH + kh) * kW + kw];
            }
          }
        }
        out[(static_cast<size_t>(oc) * outH + oh) * outW + ow] = acc;
      }
    }
  }
}

}  // namespace infercore
