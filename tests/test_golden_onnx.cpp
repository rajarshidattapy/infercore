// Golden test: the native backend (fp32) must produce the same outputs as the
// ONNX Runtime backend running the exported model, within a tight tolerance.
#include <cmath>
#include <fstream>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "infercore/model.hpp"

namespace {

const char* kGraph = "models/mnist_cnn.json";
const char* kFp32 = "models/mnist_fp32.weights";
const char* kOnnx = "models/mnist_cnn.onnx";

bool files_exist() {
  for (const char* p : {kGraph, kFp32, kOnnx}) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
  }
  return true;
}

TEST(GoldenOnnx, NativeMatchesOnnxRuntime) {
#ifdef INFERCORE_WITH_ONNX
  if (!files_exist())
    GTEST_SKIP() << "model files not generated yet (run tools/train_model.py)";
  auto native = infercore::load_native_model(kGraph, kFp32, infercore::Precision::kFp32, 1);
  auto onnx = infercore::load_onnx_model(kOnnx);

  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float worst = 0.0f;
  for (int trial = 0; trial < 5; ++trial) {
    std::vector<float> pixels(784);
    for (auto& p : pixels) p = dist(rng);
    infercore::Tensor in({1, 1, 28, 28}, pixels);
    infercore::Tensor a = native->run(in);
    infercore::Tensor b = onnx->run(in);
    ASSERT_EQ(a.numel(), b.numel());
    for (size_t i = 0; i < a.numel(); ++i)
      worst = std::max(worst, std::fabs(a[i] - b[i]));
  }
  std::printf("native vs ONNX Runtime max abs diff over 5 inputs: %.3e\n", worst);
  EXPECT_LT(worst, 1e-4f);
#else
  GTEST_SKIP() << "built without ONNX Runtime support";
#endif
}

}  // namespace
