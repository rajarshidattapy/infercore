/* InferCore C ABI - load a model, run inference, free.
 *
 * Shared-library consumers: build libinfercore (.so/.dll), then either link
 * against it or resolve these symbols at runtime (see demo/demo.cpp, which
 * uses dlopen/LoadLibrary).
 */
#ifndef INFERCORE_H
#define INFERCORE_H

#include <stdint.h>

#if defined(INFERCORE_STATIC)
#  define IC_API
#elif defined(_WIN32)
#  if defined(INFERCORE_BUILD)
#    define IC_API __declspec(dllexport)
#  else
#    define IC_API __declspec(dllimport)
#  endif
#else
#  define IC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ic_model ic_model;

/* Library version string, e.g. "0.1.0". */
IC_API const char* ic_version(void);

/* Load a native model. precision: "fp32" or "int8".
 * Returns NULL on failure; call ic_last_error() for the reason. */
IC_API ic_model* ic_load(const char* graph_json_path, const char* weights_path,
                         const char* precision);

/* Load an ONNX model through the ONNX Runtime backend (fp32).
 * Returns NULL on failure; call ic_last_error() for the reason. */
IC_API ic_model* ic_load_onnx(const char* onnx_path);

/* Number of float elements the model's input / output expects. */
IC_API int64_t ic_input_size(const ic_model* model);
IC_API int64_t ic_output_size(const ic_model* model);

/* Run inference. input/output are flat float buffers in the model's layout.
 * Returns 0 on success, non-zero on failure (see ic_last_error()). */
IC_API int ic_run(ic_model* model, const float* input, int64_t input_len, float* output,
                  int64_t output_len);

/* Last error for this thread, or an empty string. */
IC_API const char* ic_last_error(void);

IC_API void ic_free(ic_model* model);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* INFERCORE_H */
