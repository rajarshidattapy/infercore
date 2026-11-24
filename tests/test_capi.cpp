#include <cstring>
#include <fstream>
#include <vector>

#include "gtest/gtest.h"
#include "infercore/infercore.h"

namespace {

TEST(CApi, Version) {
  const char* v = ic_version();
  ASSERT_NE(v, nullptr);
  EXPECT_GT(std::strlen(v), 0u);
}

TEST(CApi, LoadFailureSetsError) {
  ic_model* m = ic_load("no/such/graph.json", "no/such.weights", "fp32");
  EXPECT_EQ(m, nullptr);
  EXPECT_GT(std::strlen(ic_last_error()), 0u);
}

TEST(CApi, FullRoundTrip) {
  std::ifstream probe("models/mnist_cnn.json");
  if (!probe) GTEST_SKIP() << "model files not generated yet (run tools/train_model.py)";
  ic_model* m = ic_load("models/mnist_cnn.json", "models/mnist_fp32.weights", "fp32");
  ASSERT_NE(m, nullptr) << ic_last_error();
  ASSERT_EQ(ic_input_size(m), 784);
  ASSERT_EQ(ic_output_size(m), 10);
  std::vector<float> in(784, 0.5f), out(10, 0.0f);
  ASSERT_EQ(ic_run(m, in.data(), 784, out.data(), 10), 0) << ic_last_error();
  float sum = 0.0f;
  for (float v : out) sum += v;
  EXPECT_NEAR(sum, 1.0f, 1e-3f);  // softmax output
  ic_free(m);
}

}  // namespace
