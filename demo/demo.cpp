// InferCore demo: loads libinfercore at RUNTIME (dlopen / LoadLibrary),
// resolves the C ABI by name, and classifies a sample MNIST digit with both
// the fp32 and the int8 model, concurrently, to show model orchestration.
//
// Usage: infercore_demo <graph.json> <fp32.weights> <int8.weights> <sample.bin>
// sample.bin: int64 label followed by 784 float32 pixel values.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
using LibHandle = HMODULE;
#else
#  include <dlfcn.h>
using LibHandle = void*;
#endif

#include "infercore/infercore.h"

namespace {

struct IcApi {
  decltype(&ic_version) version;
  decltype(&ic_load) load;
  decltype(&ic_load_onnx) load_onnx;
  decltype(&ic_input_size) input_size;
  decltype(&ic_output_size) output_size;
  decltype(&ic_run) run;
  decltype(&ic_last_error) last_error;
  decltype(&ic_free) free;
};

LibHandle open_lib(const char* path) {
#if defined(_WIN32)
  return LoadLibraryA(path);
#else
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* sym(LibHandle lib, const char* name) {
#if defined(_WIN32)
  void* p = reinterpret_cast<void*>(GetProcAddress(lib, name));
#else
  void* p = dlsym(lib, name);
#endif
  return p;
}

}  // namespace

static void* sym_checked(LibHandle lib, const char* name) {
  void* p = sym(lib, name);
  if (!p) {
    std::fprintf(stderr, "missing symbol %s\n", name);
    std::exit(1);
  }
  return p;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <graph.json> <fp32.weights> <int8.weights> <sample.bin>\n",
                 argv[0]);
    return 1;
  }

#if defined(_WIN32)
  const char* lib_path = "infercore.dll";
#else
  const char* lib_path = "./libinfercore.so";
#endif
  if (const char* env = std::getenv("INFERCORE_LIB")) lib_path = env;

  LibHandle lib = open_lib(lib_path);
  if (!lib) {
#if defined(_WIN32)
    std::fprintf(stderr, "LoadLibrary failed for %s (error %lu)\n", lib_path,
                 GetLastError());
#else
    std::fprintf(stderr, "dlopen failed for %s: %s\n", lib_path, dlerror());
#endif
    return 1;
  }
  std::printf("loaded runtime library: %s\n", lib_path);

  IcApi ic;
  ic.version = reinterpret_cast<decltype(ic.version)>(sym_checked(lib, "ic_version"));
  ic.load = reinterpret_cast<decltype(ic.load)>(sym_checked(lib, "ic_load"));
  ic.load_onnx = reinterpret_cast<decltype(ic.load_onnx)>(sym_checked(lib, "ic_load_onnx"));
  ic.input_size = reinterpret_cast<decltype(ic.input_size)>(sym_checked(lib, "ic_input_size"));
  ic.output_size = reinterpret_cast<decltype(ic.output_size)>(sym_checked(lib, "ic_output_size"));
  ic.run = reinterpret_cast<decltype(ic.run)>(sym_checked(lib, "ic_run"));
  ic.last_error = reinterpret_cast<decltype(ic.last_error)>(sym_checked(lib, "ic_last_error"));
  ic.free = reinterpret_cast<decltype(ic.free)>(sym_checked(lib, "ic_free"));
  std::printf("InferCore runtime version %s (via C ABI)\n", ic.version());

  // Read the sample: int64 label + 784 float32 pixels.
  FILE* f = std::fopen(argv[4], "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open sample %s\n", argv[4]);
    return 1;
  }
  int64_t label = -1;
  std::fread(&label, sizeof(label), 1, f);
  std::vector<float> pixels(784);
  std::fread(pixels.data(), sizeof(float), pixels.size(), f);
  std::fclose(f);
  std::printf("sample label: %lld\n", static_cast<long long>(label));

  // Orchestration: run the fp32 and int8 models concurrently.
  struct Result {
    int top1 = -1;
    float top1_conf = 0.0f;
    bool ok = false;
  };
  auto classify = [&](const char* precision, Result& res) {
    ic_model* m = ic.load(argv[1], precision[0] == 'i' ? argv[3] : argv[2], precision);
    if (!m) {
      std::fprintf(stderr, "[%s] ic_load failed: %s\n", precision, ic.last_error());
      return;
    }
    const int64_t n_out = ic.output_size(m);
    std::vector<float> out(static_cast<size_t>(n_out));
    if (ic.run(m, pixels.data(), static_cast<int64_t>(pixels.size()), out.data(),
               n_out) != 0) {
      std::fprintf(stderr, "[%s] ic_run failed: %s\n", precision, ic.last_error());
      ic.free(m);
      return;
    }
    int best = 0;
    for (int64_t i = 1; i < n_out; ++i)
      if (out[i] > out[best]) best = static_cast<int>(i);
    res.top1 = best;
    res.top1_conf = out[best];
    res.ok = true;
    ic.free(m);
  };

  Result r32, r8;
  std::thread t32([&] { classify("fp32", r32); });
  std::thread t8([&] { classify("int8", r8); });
  t32.join();
  t8.join();

  if (r32.ok)
    std::printf("fp32 prediction: %d (confidence %.3f) %s\n", r32.top1, r32.top1_conf,
                r32.top1 == label ? "CORRECT" : "WRONG");
  if (r8.ok)
    std::printf("int8 prediction: %d (confidence %.3f) %s\n", r8.top1, r8.top1_conf,
                r8.top1 == label ? "CORRECT" : "WRONG");
  return (r32.ok && r8.ok) ? 0 : 1;
}
