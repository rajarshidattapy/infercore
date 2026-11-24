#include "infercore/infercore.h"

#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <thread>

#include "infercore/model.hpp"

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }

infercore::Precision parse_precision(const char* p) {
  if (p && std::strcmp(p, "int8") == 0) return infercore::Precision::kInt8;
  return infercore::Precision::kFp32;
}

}  // namespace

struct ic_model {
  std::unique_ptr<infercore::IModel> impl;
};

extern "C" {

IC_API const char* ic_version(void) { return "0.1.0"; }

IC_API ic_model* ic_load(const char* graph_json_path, const char* weights_path,
                         const char* precision) {
  if (!graph_json_path || !weights_path) {
    set_error("ic_load: null path");
    return nullptr;
  }
  try {
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0 || threads > 4) threads = 4;
    auto m = std::make_unique<ic_model>();
    m->impl = infercore::load_native_model(graph_json_path, weights_path,
                                           parse_precision(precision), threads);
    return m.release();
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  }
}

IC_API ic_model* ic_load_onnx(const char* onnx_path) {
  if (!onnx_path) {
    set_error("ic_load_onnx: null path");
    return nullptr;
  }
  try {
    auto m = std::make_unique<ic_model>();
    m->impl = infercore::load_onnx_model(onnx_path);
    return m.release();
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  }
}

IC_API int64_t ic_input_size(const ic_model* model) {
  if (!model) return -1;
  int64_t n = 1;
  for (int64_t d : model->impl->input_shape()) n *= d;
  return n;
}

IC_API int64_t ic_output_size(const ic_model* model) {
  if (!model) return -1;
  int64_t n = 1;
  for (int64_t d : model->impl->output_shape()) n *= d;
  return n;
}

IC_API int ic_run(ic_model* model, const float* input, int64_t input_len, float* output,
                  int64_t output_len) {
  if (!model || !input || !output) {
    set_error("ic_run: null argument");
    return 1;
  }
  try {
    infercore::Tensor in(model->impl->input_shape());
    if (static_cast<int64_t>(in.numel()) != input_len) {
      set_error("ic_run: input_len mismatch");
      return 2;
    }
    std::memcpy(in.data(), input, in.numel() * sizeof(float));
    infercore::Tensor out = model->impl->run(in);
    if (static_cast<int64_t>(out.numel()) > output_len) {
      set_error("ic_run: output buffer too small");
      return 3;
    }
    std::memcpy(output, out.data(), out.numel() * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    set_error(e.what());
    return 4;
  }
}

IC_API const char* ic_last_error(void) { return g_last_error.c_str(); }

IC_API void ic_free(ic_model* model) { delete model; }

}  // extern "C"
