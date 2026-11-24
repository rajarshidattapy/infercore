// InferCore - native compute kernels (fp32 and int8).
#pragma once

#include <cstddef>
#include <cstdint>

namespace infercore {

class ThreadPool;

// Row-major GEMM.
//   transB = false: C[M,N] = A[M,K] * B[K,N]
//   transB = true : C[M,N] = A[M,K] * B[N,K]^T   (B rows are output channels)
// When `pool` is non-null, rows of C are computed in parallel.
void gemm_f32(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
              bool transB, ThreadPool* pool = nullptr);

// int8 x int8 -> int32 GEMM with B in transposed ([N,K]) layout so both inner
// loops walk contiguous memory. Accumulation is exact int32.
void gemm_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C, size_t M, size_t N,
                  size_t K, ThreadPool* pool = nullptr);

// im2col for NCHW input with a single image (N = 1).
// Output col has layout [outH*outW, C*kH*kW] (one patch per row).
void im2col_f32(const float* in, int C, int H, int W, int kH, int kW, int stride,
                int pad, int outH, int outW, float* col);
void im2col_s8(const int8_t* in, int C, int H, int W, int kH, int kW, int stride,
               int pad, int outH, int outW, int8_t* col);

// Direct (naive) fp32 convolution, used as the reference in unit tests.
void conv2d_f32_reference(const float* in, const float* w, const float* bias, int C, int H,
                          int Wd, int OC, int kH, int kW, int stride, int pad,
                          float* out);

}  // namespace infercore
