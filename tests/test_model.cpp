#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "gtest/gtest.h"
#include "infercore/model.hpp"

namespace {

const char* kGraph = "models/mnist_cnn.json";
const char* kFp32 = "models/mnist_fp32.weights";
const char* kInt8 = "models/mnist_int8.weights";
const char* kSamples = "models/samples.bin";  // 10 x (int64 label + 784 f32)

bool files_exist() {
  for (const char* p : {kGraph, kFp32, kInt8, kSamples}) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
  }
  return true;
}

struct Sample {
  int64_t label;
  std::vector<float> pixels;
};

std::vector<Sample> load_samples() {
  std::ifstream f(kSamples, std::ios::binary);
  std::vector<Sample> out;
  for (;;) {
    Sample s;
    s.pixels.resize(784);
    f.read(reinterpret_cast<char*>(&s.label), 8);
    if (!f) break;
    f.read(reinterpret_cast<char*>(s.pixels.data()), 784 * 4);
    if (!f) break;
    out.push_back(std::move(s));
  }
  return out;
}

int top1(const infercore::Tensor& t) {
  int best = 0;
  for (size_t i = 1; i < t.numel(); ++i)
    if (t[i] > t[best]) best = static_cast<int>(i);
  return best;
}

TEST(NativeFp32, ClassifiesSamples) {
  if (!files_exist())
    GTEST_SKIP() << "model files not generated yet (run tools/train_model.py)";
  auto model = infercore::load_native_model(kGraph, kFp32, infercore::Precision::kFp32, 2);
  int correct = 0;
  const auto samples = load_samples();
  ASSERT_EQ(samples.size(), 10u);
  for (const auto& s : samples) {
    infercore::Tensor in({1, 1, 28, 28}, s.pixels);
    if (top1(model->run(in)) == s.label) ++correct;
  }
  // Trained model is well above this on these fixed samples; threshold keeps
  // the test robust across rebuilds while still proving the pipeline works.
  EXPECT_GE(correct, 8);
}

TEST(NativeInt8, AgreesWithFp32OnSamples) {
  if (!files_exist())
    GTEST_SKIP() << "model files not generated yet (run tools/train_model.py)";
  auto m32 = infercore::load_native_model(kGraph, kFp32, infercore::Precision::kFp32, 2);
  auto m8 = infercore::load_native_model(kGraph, kInt8, infercore::Precision::kInt8, 2);
  int agree = 0;
  const auto samples = load_samples();
  for (const auto& s : samples) {
    infercore::Tensor in({1, 1, 28, 28}, s.pixels);
    if (top1(m32->run(in)) == top1(m8->run(in))) ++agree;
  }
  EXPECT_GE(agree, 9);  // PTQ on this model keeps top-1 on the sample set
}

TEST(NativeInt8, OutputSumsToOne) {
  if (!files_exist())
    GTEST_SKIP() << "model files not generated yet (run tools/train_model.py)";
  auto m8 = infercore::load_native_model(kGraph, kInt8, infercore::Precision::kInt8, 2);
  const auto samples = load_samples();
  infercore::Tensor in({1, 1, 28, 28}, samples[0].pixels);
  infercore::Tensor out = m8->run(in);
  float sum = 0.0f;
  for (size_t i = 0; i < out.numel(); ++i) sum += out[i];
  EXPECT_NEAR(sum, 1.0f, 1e-2f);
}

TEST(NativeModel, PerOpTimingsRecorded) {
  if (!files_exist())
    GTEST_SKIP() << "model files not generated yet (run tools/train_model.py)";
  auto model = infercore::load_native_model(kGraph, kFp32, infercore::Precision::kFp32, 2);
  const auto samples = load_samples();
  infercore::Tensor in({1, 1, 28, 28}, samples[0].pixels);
  model->run(in);
  const auto& timings = model->last_timings();
  EXPECT_EQ(timings.size(), 9u);  // conv,relu,pool,conv,relu,pool,flatten,gemm,softmax
  for (const auto& t : timings) EXPECT_GE(t.micros, 0.0);
}

}  // namespace
