#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "infercore/kernels.hpp"
#include "infercore/tensor.hpp"

namespace {

std::vector<float> rand_vec(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

TEST(Gemm, TransBMatchesNaive) {
  const size_t M = 5, N = 7, K = 11;
  const auto A = rand_vec(M * K, 42);
  const auto B = rand_vec(N * K, 43);  // [N,K], used transposed
  std::vector<float> C(M * N, 0.0f);
  infercore::gemm_f32(A.data(), B.data(), C.data(), M, N, K, /*transB=*/true);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      float ref = 0.0f;
      for (size_t k = 0; k < K; ++k) ref += A[m * K + k] * B[n * K + k];
      EXPECT_NEAR(C[m * N + n], ref, 1e-4f) << "m=" << m << " n=" << n;
    }
}

TEST(Gemm, NoTransMatchesNaive) {
  const size_t M = 4, N = 6, K = 9;
  const auto A = rand_vec(M * K, 7);
  const auto B = rand_vec(K * N, 8);  // [K,N]
  std::vector<float> C(M * N, 0.0f);
  infercore::gemm_f32(A.data(), B.data(), C.data(), M, N, K, /*transB=*/false);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      float ref = 0.0f;
      for (size_t k = 0; k < K; ++k) ref += A[m * K + k] * B[k * N + n];
      EXPECT_NEAR(C[m * N + n], ref, 1e-4f);
    }
}

TEST(GemmInt8, MatchesInt32Reference) {
  const size_t M = 6, N = 5, K = 40;  // K > 8 to exercise the unrolled path
  std::vector<int8_t> A(M * K), B(N * K);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127);
  for (auto& x : A) x = static_cast<int8_t>(dist(rng));
  for (auto& x : B) x = static_cast<int8_t>(dist(rng));
  std::vector<int32_t> C(M * N, 0);
  infercore::gemm_s8s8s32(A.data(), B.data(), C.data(), M, N, K);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      int32_t ref = 0;
      for (size_t k = 0; k < K; ++k)
        ref += static_cast<int32_t>(A[m * K + k]) * B[n * K + k];
      EXPECT_EQ(C[m * N + n], ref);
    }
}

TEST(Im2col, LayoutMatchesIndexing) {
  // 1x3x3 input, values 1..9; 2x2 kernel, stride 1, pad 0 -> 2x2 patches.
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> col(4 * 4, 0.0f);
  infercore::im2col_f32(in.data(), 1, 3, 3, 2, 2, 1, 0, 2, 2, col.data());
  // patch (0,0): [1,2,4,5]
  EXPECT_FLOAT_EQ(col[0], 1);
  EXPECT_FLOAT_EQ(col[1], 2);
  EXPECT_FLOAT_EQ(col[2], 4);
  EXPECT_FLOAT_EQ(col[3], 5);
  // patch (1,1): [5,6,8,9]
  EXPECT_FLOAT_EQ(col[3 * 4 + 0], 5);
  EXPECT_FLOAT_EQ(col[3 * 4 + 3], 9);
}

TEST(Im2col, ZeroPadding) {
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // 3x3 kernel, pad 1 -> outH=outW=3, corner patch reads 4 in-bounds values.
  std::vector<float> col(9 * 9, -1.0f);
  infercore::im2col_f32(in.data(), 1, 3, 3, 3, 3, 1, 1, 3, 3, col.data());
  // patch (0,0) center-bottom-right 2x2 = [0,0,0,1, 0,0,2... ] check known cells:
  // rows: kh=0 -> ih=-1 all zero; kh=1 -> ih=0: iw=-1->0, 1, 2; kh=2 -> ih=1: 0,4,5
  const float* p00 = col.data();
  EXPECT_FLOAT_EQ(p00[0], 0); EXPECT_FLOAT_EQ(p00[1], 0); EXPECT_FLOAT_EQ(p00[2], 0);
  EXPECT_FLOAT_EQ(p00[3], 0); EXPECT_FLOAT_EQ(p00[4], 1); EXPECT_FLOAT_EQ(p00[5], 2);
  EXPECT_FLOAT_EQ(p00[6], 0); EXPECT_FLOAT_EQ(p00[7], 4); EXPECT_FLOAT_EQ(p00[8], 5);
}

TEST(Quant, RoundTripErrorBounded) {
  const auto v = rand_vec(1000, 99);
  infercore::Tensor t({1000}, v);
  infercore::QTensor q = infercore::quantize_dynamic(t);
  infercore::Tensor back = infercore::dequantize(q);
  float absmax = 0.0f;
  for (float x : v) absmax = std::max(absmax, std::fabs(x));
  const float tol = absmax / 127.0f;  // one quantization step
  for (size_t i = 0; i < v.size(); ++i)
    EXPECT_NEAR(back[i], v[i], tol) << "i=" << i;
}

TEST(Quant, PerChannelScales) {
  // channel 0 small values, channel 1 large: per-channel scales must differ.
  infercore::Tensor w({2, 4}, {0.01f, -0.01f, 0.02f, 0.01f, 10.f, -10.f, 5.f, 2.f});
  infercore::QWeights q = infercore::quantize_weights_per_channel(w);
  ASSERT_EQ(q.scales.size(), 2u);
  EXPECT_NEAR(q.scales[0], 0.02f / 127.0f, 1e-9f);
  EXPECT_NEAR(q.scales[1], 10.0f / 127.0f, 1e-9f);
}

}  // namespace
